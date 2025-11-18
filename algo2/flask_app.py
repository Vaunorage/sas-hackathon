"""
Flask API for GPU-based Actuarial Projections
Provides RESTful endpoints for job management and result retrieval
"""

import os
import sqlite3
import json
import threading
import traceback
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any
import pandas as pd

# Load environment variables from .env file if it exists
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed, continue without it

from flask import Flask, request, jsonify, send_file, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename

# PostgreSQL support
try:
    import psycopg
    from psycopg.rows import dict_row
    POSTGRES_AVAILABLE = True
except ImportError:
    print("Warning: psycopg not available. Install it to use NeonDB/PostgreSQL support.")
    POSTGRES_AVAILABLE = False

# Import the GPU projection function
try:
    from gpu import run_projection_gpu
    GPU_AVAILABLE = True
except Exception as e:
    print(f"Warning: GPU module not available: {e}")
    GPU_AVAILABLE = False

# =============================================================================
# CONFIGURATION
# =============================================================================

app = Flask(__name__, static_folder='static')
CORS(app)  # Enable CORS for all routes
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max upload
app.config['UPLOAD_FOLDER'] = Path(__file__).parent / 'uploads'
app.config['RESULTS_FOLDER'] = Path(__file__).parent / 'results'
app.config['DATABASE'] = Path(__file__).parent / 'jobs.db'
app.config['DEFAULT_DATA_FOLDER'] = Path(__file__).parent / 'data_in'

# Create directories if they don't exist
app.config['UPLOAD_FOLDER'].mkdir(exist_ok=True)
app.config['RESULTS_FOLDER'].mkdir(exist_ok=True)
Path(__file__).parent.joinpath('static').mkdir(exist_ok=True)

# Application metadata
APP_VERSION = "1.0.0"
ENVIRONMENT = os.getenv('ENVIRONMENT', 'development')
ADMIN_PASSWORD = os.getenv('ADMIN_PASSWORD', 'admin123')  # Change in production!

# Database configuration
USE_NEONDB = os.getenv('USE_NEONDB', 'false').lower() == 'true'
NEONDB_URL = os.getenv('NEONDB_URL', 'postgresql://neondb_owner:npg_U8nuV5Zzbsge@ep-spring-hall-a448t160-pooler.us-east-1.aws.neon.tech/neondb?sslmode=require&channel_binding=require')

# Allowed file extensions
ALLOWED_EXTENSIONS = {'csv'}

# =============================================================================
# DATABASE INITIALIZATION
# =============================================================================

def init_db():
    """Initialize the database (SQLite or PostgreSQL) with jobs table"""
    if USE_NEONDB:
        if not POSTGRES_AVAILABLE:
            raise Exception("PostgreSQL support requires psycopg. Install it with: uv add psycopg[binary]")
        
        conn = psycopg.connect(NEONDB_URL)
        cursor = conn.cursor()
        
        # PostgreSQL syntax
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS jobs (
                job_id TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL,
                started_at TEXT,
                completed_at TEXT,
                error_message TEXT,
                parameters TEXT,
                uploaded_files TEXT,
                result_files TEXT,
                current_batch INTEGER DEFAULT 0,
                total_batches INTEGER DEFAULT 0,
                progress_percent REAL DEFAULT 0.0
            )
        """)
        
        conn.commit()
        cursor.close()
        conn.close()
    else:
        # SQLite initialization
        conn = sqlite3.connect(app.config['DATABASE'])
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS jobs (
                job_id TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL,
                started_at TEXT,
                completed_at TEXT,
                error_message TEXT,
                parameters TEXT,
                uploaded_files TEXT,
                result_files TEXT,
                current_batch INTEGER DEFAULT 0,
                total_batches INTEGER DEFAULT 0,
                progress_percent REAL DEFAULT 0.0
            )
        """)
        
        # Migrate existing database to add new columns if they don't exist
        try:
            cursor.execute("SELECT current_batch FROM jobs LIMIT 1")
        except sqlite3.OperationalError:
            # Column doesn't exist, add it
            print("Migrating database: Adding progress tracking columns...")
            cursor.execute("ALTER TABLE jobs ADD COLUMN current_batch INTEGER DEFAULT 0")
            cursor.execute("ALTER TABLE jobs ADD COLUMN total_batches INTEGER DEFAULT 0")
            cursor.execute("ALTER TABLE jobs ADD COLUMN progress_percent REAL DEFAULT 0.0")
            print("Migration complete")
        
        conn.commit()
        conn.close()

# Initialize database on startup
init_db()

# Track if database is ready
db_initialized = True

# Track running jobs and their threads
job_threads = {}
job_cancellation_flags = {}
job_progress = {}  # In-memory progress tracking: {job_id: {'current': int, 'total': int}}

# =============================================================================
# DATABASE HELPERS
# =============================================================================

def get_db_connection():
    """Get a database connection (SQLite or PostgreSQL)"""
    if USE_NEONDB:
        if not POSTGRES_AVAILABLE:
            raise Exception("PostgreSQL support requires psycopg. Install it with: uv add psycopg[binary]")
        return psycopg.connect(NEONDB_URL, row_factory=dict_row)
    else:
        conn = sqlite3.connect(app.config['DATABASE'])
        conn.row_factory = sqlite3.Row
        return conn

def create_job(job_id: str, parameters: Dict[str, Any], uploaded_files: list) -> None:
    """Create a new job in the database"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Use %s for PostgreSQL, ? for SQLite
    placeholder = '%s' if USE_NEONDB else '?'
    
    cursor.execute(f"""
        INSERT INTO jobs (job_id, status, created_at, parameters, uploaded_files)
        VALUES ({placeholder}, {placeholder}, {placeholder}, {placeholder}, {placeholder})
    """, (
        job_id,
        'pending',
        datetime.utcnow().isoformat(),
        json.dumps(parameters),
        json.dumps(uploaded_files)
    ))
    
    conn.commit()
    cursor.close()
    conn.close()

def update_job_status(job_id: str, status: str, error_message: Optional[str] = None) -> None:
    """Update job status"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    updates = {'status': status}
    
    if status == 'running' and error_message is None:
        updates['started_at'] = datetime.utcnow().isoformat()
    elif status in ['completed', 'failed']:
        updates['completed_at'] = datetime.utcnow().isoformat()
    
    if error_message:
        updates['error_message'] = error_message
    
    placeholder = '%s' if USE_NEONDB else '?'
    set_clause = ', '.join([f"{k} = {placeholder}" for k in updates.keys()])
    values = list(updates.values()) + [job_id]
    
    cursor.execute(f"UPDATE jobs SET {set_clause} WHERE job_id = {placeholder}", values)
    conn.commit()
    cursor.close()
    conn.close()

def update_job_progress(job_id: str, current_batch: int, total_batches: int) -> None:
    """
    Update job progress
    
    Args:
        job_id: Job identifier
        current_batch: Current batch number (1-indexed)
        total_batches: Total number of batches
    """
    progress_percent = (current_batch / total_batches * 100.0) if total_batches > 0 else 0.0
    
    print(f"  Progress update: {job_id} - Batch {current_batch}/{total_batches} ({progress_percent:.1f}%)")
    
    # Update in-memory progress for fast access
    job_progress[job_id] = {
        'current': current_batch,
        'total': total_batches,
        'percent': progress_percent
    }
    
    # Update database
    conn = get_db_connection()
    cursor = conn.cursor()
    
    placeholder = '%s' if USE_NEONDB else '?'
    
    cursor.execute(f"""
        UPDATE jobs 
        SET current_batch = {placeholder}, total_batches = {placeholder}, progress_percent = {placeholder}
        WHERE job_id = {placeholder}
    """, (current_batch, total_batches, progress_percent, job_id))
    
    conn.commit()
    cursor.close()
    conn.close()

def update_job_results(job_id: str, result_files: list) -> None:
    """
    Update job with result files
    
    Args:
        job_id: Job identifier
        result_files: List of file dictionaries with keys: name, type, description, size
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    
    placeholder = '%s' if USE_NEONDB else '?'
    
    cursor.execute(f"""
        UPDATE jobs SET result_files = {placeholder} WHERE job_id = {placeholder}
    """, (json.dumps(result_files), job_id))
    
    conn.commit()
    cursor.close()
    conn.close()

def get_job(job_id: str) -> Optional[Dict[str, Any]]:
    """Get job details by ID"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    placeholder = '%s' if USE_NEONDB else '?'
    
    cursor.execute(f"SELECT * FROM jobs WHERE job_id = {placeholder}", (job_id,))
    row = cursor.fetchone()
    cursor.close()
    conn.close()
    
    if row:
        # Handle both SQLite Row objects and PostgreSQL dict rows
        # Both should work as dict-like objects now
        row_dict = dict(row)
        
        job_data = {
            'job_id': row_dict['job_id'],
            'status': row_dict['status'],
            'created_at': row_dict['created_at'],
            'started_at': row_dict['started_at'],
            'completed_at': row_dict['completed_at'],
            'error_message': row_dict['error_message'],
            'parameters': json.loads(row_dict['parameters']) if row_dict['parameters'] else {},
            'uploaded_files': json.loads(row_dict['uploaded_files']) if row_dict['uploaded_files'] else [],
            'result_files': json.loads(row_dict['result_files']) if row_dict['result_files'] else []
        }
        
        # Handle progress fields (may not exist in old records)
        try:
            job_data['current_batch'] = row_dict['current_batch'] if row_dict['current_batch'] is not None else 0
            job_data['total_batches'] = row_dict['total_batches'] if row_dict['total_batches'] is not None else 0
            job_data['progress_percent'] = row_dict['progress_percent'] if row_dict['progress_percent'] is not None else 0.0
        except (KeyError, IndexError):
            job_data['current_batch'] = 0
            job_data['total_batches'] = 0
            job_data['progress_percent'] = 0.0
        
        return job_data
    return None

def get_all_jobs() -> list:
    """Get all jobs ordered by creation date"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("SELECT * FROM jobs ORDER BY created_at DESC")
    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    
    jobs = []
    for row in rows:
        # Handle both SQLite Row objects and PostgreSQL dict rows
        # Both should work as dict-like objects now
        row_dict = dict(row)
        
        job_data = {
            'job_id': row_dict['job_id'],
            'status': row_dict['status'],
            'created_at': row_dict['created_at'],
            'started_at': row_dict['started_at'],
            'completed_at': row_dict['completed_at'],
            'error_message': row_dict['error_message'],
            'parameters': json.loads(row_dict['parameters']) if row_dict['parameters'] else {},
            'uploaded_files': json.loads(row_dict['uploaded_files']) if row_dict['uploaded_files'] else [],
            'result_files': json.loads(row_dict['result_files']) if row_dict['result_files'] else []
        }
        
        # Handle progress fields (may not exist in old records)
        try:
            job_data['current_batch'] = row_dict['current_batch'] if row_dict['current_batch'] is not None else 0
            job_data['total_batches'] = row_dict['total_batches'] if row_dict['total_batches'] is not None else 0
            job_data['progress_percent'] = row_dict['progress_percent'] if row_dict['progress_percent'] is not None else 0.0
        except (KeyError, IndexError):
            job_data['current_batch'] = 0
            job_data['total_batches'] = 0
            job_data['progress_percent'] = 0.0
        
        jobs.append(job_data)
    return jobs

# =============================================================================
# FILE HELPERS
# =============================================================================

def allowed_file(filename: str) -> bool:
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_job_upload_folder(job_id: str) -> Path:
    """Get upload folder for a specific job"""
    folder = app.config['UPLOAD_FOLDER'] / job_id
    folder.mkdir(exist_ok=True)
    return folder

def get_job_results_folder(job_id: str) -> Path:
    """Get results folder for a specific job"""
    folder = app.config['RESULTS_FOLDER'] / job_id
    folder.mkdir(exist_ok=True)
    return folder

# =============================================================================
# JOB PROCESSING
# =============================================================================

def process_job(job_id: str):
    """
    Process a job in the background
    This runs the GPU projection with the uploaded data
    """
    try:
        # Check if job was cancelled before starting
        if job_cancellation_flags.get(job_id, False):
            update_job_status(job_id, 'cancelled')
            return
        
        # Update status to running
        update_job_status(job_id, 'running')
        
        # Get job details
        job = get_job(job_id)
        if not job:
            raise Exception(f"Job {job_id} not found")
        
        # Check again after getting job details
        if job_cancellation_flags.get(job_id, False):
            update_job_status(job_id, 'cancelled')
            return
        
        # Get parameters
        params = job['parameters']
        nb_years = params.get('nb_an_projection', 100)
        nb_scenarios = params.get('nb_scenarios', 100)
        max_accounts = params.get('max_accounts', None)
        debug_account = params.get('debug_account', None)
        
        # Set up paths
        # Check if job has uploaded files
        upload_folder = get_job_upload_folder(job_id)
        has_uploaded_files = job.get('uploaded_files') and len(job['uploaded_files']) > 0
        
        # Use uploaded files folder if files were uploaded, otherwise use default data folder
        if has_uploaded_files:
            data_path = upload_folder
        else:
            data_path = app.config['DEFAULT_DATA_FOLDER']
        
        output_path = get_job_results_folder(job_id)
        
        # Get custom file paths for mixed uploads (some uploaded, some from defaults)
        default_data_path = app.config['DEFAULT_DATA_FOLDER']
        uploaded_file_list = job.get('uploaded_files', [])
        
        custom_paths = {}
        
        # If files were uploaded, specify individual paths for each file
        if has_uploaded_files:
            file_mapping = {
                'population_path': 'POPULATION.csv',
                'mortalite_path': 'MORTALITE.csv',
                'rendements_path': 'RENDEMENTS.csv',
                'depots_futurs_path': 'DEPOTS_FUTURS.csv',
                'frais_admin_path': 'FRAIS_ADMIN.csv',
                'min_ferr_path': 'MIN_FERR.csv',
                'tx_lapse_part_path': 'TX_LAPSE_PART.csv',
                'tx_lapse_tot_path': 'TX_LAPSE_TOT.csv',
                'acquisition_path': 'ACQUISITION.csv',
                'coussins_escap_path': 'COUSSINS_ESCAP.csv'
            }
            
            for path_key, filename in file_mapping.items():
                if filename in uploaded_file_list:
                    # Use uploaded file
                    custom_paths[path_key] = upload_folder / filename
                else:
                    # Use default file
                    custom_paths[path_key] = default_data_path / filename
        
        # Define progress callback
        def progress_callback(current_batch: int, total_batches: int):
            """Callback to update job progress"""
            update_job_progress(job_id, current_batch, total_batches)
        
        # Run GPU projection
        print(f"Starting GPU projection for job {job_id}")
        print(f"  Data path: {data_path}")
        print(f"  Uploaded files: {uploaded_file_list}")
        if custom_paths:
            print(f"  Using custom paths for {len(custom_paths)} files")
        results = run_projection_gpu(
            data_path=data_path,
            output_path=output_path,
            nb_an_projection=nb_years,
            nb_scenarios=nb_scenarios,
            max_accounts=max_accounts,
            debug_account=debug_account,
            progress_callback=progress_callback,
            **custom_paths
        )
        
        # List result files with metadata
        result_files = []
        if output_path.exists():
            # Define expected output files with their metadata
            output_file_metadata = {
                'FLUX_PROJETES_GPU.csv': {
                    'type': 'internal',
                    'description': 'Projected cash flows by time period (year/month)'
                },
                'VP_FLUX_COMPTE_GPU.csv': {
                    'type': 'detailed',
                    'description': 'Present values by account'
                },
                'VP_FLUX_TOTAL_GPU.csv': {
                    'type': 'summary',
                    'description': 'Total present value across all accounts'
                }
            }
            
            for file in output_path.glob('*.csv'):
                file_info = {
                    'name': file.name,
                    'size': file.stat().st_size
                }
                
                # Check if it's a known output file
                if file.name in output_file_metadata:
                    file_info.update(output_file_metadata[file.name])
                # Check if it's a debug file
                elif file.name.startswith('DEBUG_account_'):
                    file_info['type'] = 'debug'
                    file_info['description'] = 'Debug trace for specific account/scenario'
                else:
                    file_info['type'] = 'other'
                    file_info['description'] = 'Additional output file'
                
                result_files.append(file_info)
        
        # Update job with results
        update_job_results(job_id, result_files)
        update_job_status(job_id, 'completed')
        
        # Log saved files
        print(f"\n✓ Job {job_id} completed successfully")
        print(f"  Saved {len(result_files)} output files:")
        for file_info in result_files:
            print(f"    - {file_info['name']} ({file_info['type']}) - {file_info['description']}")
        
    except Exception as e:
        error_msg = f"{str(e)}\n{traceback.format_exc()}"
        print(f"Job {job_id} failed: {error_msg}")
        update_job_status(job_id, 'failed', error_message=error_msg)
    finally:
        # Clean up thread tracking
        if job_id in job_threads:
            del job_threads[job_id]
        if job_id in job_cancellation_flags:
            del job_cancellation_flags[job_id]
        if job_id in job_progress:
            del job_progress[job_id]

# =============================================================================
# API ROUTES - HEALTH & STATUS
# =============================================================================

@app.route('/', methods=['GET'])
def welcome():
    """Welcome endpoint with API information"""
    return jsonify({
        'message': 'GPU Actuarial Projections API',
        'version': APP_VERSION,
        'environment': ENVIRONMENT,
        'gpu_available': GPU_AVAILABLE,
        'database': 'NeonDB (PostgreSQL)' if USE_NEONDB else 'SQLite',
        'endpoints': {
            'health': '/ping',
            'ready': '/ready',
            'jobs': '/jobs',
            'create_job': 'POST /jobs',
            'get_job': '/jobs/<job_id>',
            'cancel_job': 'DELETE /jobs/<job_id>',
            'get_results': '/jobs/<job_id>/results',
            'get_files': '/jobs/<job_id>/files',
            'download_file': '/jobs/<job_id>/files/<file_name>',
            'clear_database': 'POST /admin/clear-database (requires password)'
        },
        'job_parameters': {
            'required': ['use_custom_paths=true (default mode)'],
            'optional': {
                'nb_an_projection': 'Number of years to project (default: 100)',
                'nb_scenarios': 'Number of scenarios (default: 100)',
                'max_accounts': 'Maximum number of accounts to process',
                'debug_account': 'Account ID for debugging'
            },
            'file_paths': {
                'note': 'Specify custom paths for files to override. Unspecified paths use defaults from data_in folder',
                'population_path': 'Custom path for POPULATION.csv',
                'mortalite_path': 'Custom path for MORTALITE.csv',
                'rendements_path': 'Custom path for RENDEMENTS.csv',
                'depots_futurs_path': 'Custom path for DEPOTS_FUTURS.csv',
                'frais_admin_path': 'Custom path for FRAIS_ADMIN.csv',
                'min_ferr_path': 'Custom path for MIN_FERR.csv',
                'tx_lapse_part_path': 'Custom path for TX_LAPSE_PART.csv',
                'tx_lapse_tot_path': 'Custom path for TX_LAPSE_TOT.csv',
                'acquisition_path': 'Custom path for ACQUISITION.csv',
                'coussins_escap_path': 'Custom path for COUSSINS_ESCAP.csv'
            }
        }
    })

@app.route('/web', methods=['GET'])
def web_interface():
    """Serve the web interface"""
    return send_from_directory('static', 'index.html')

@app.route('/ping', methods=['GET'])
def ping():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat()
    })

@app.route('/ready', methods=['GET'])
def ready():
    """Readiness probe - checks if database is initialized"""
    if db_initialized:
        return jsonify({
            'status': 'ready',
            'database': 'initialized',
            'database_type': 'NeonDB (PostgreSQL)' if USE_NEONDB else 'SQLite',
            'gpu_available': GPU_AVAILABLE
        })
    else:
        return jsonify({
            'status': 'not_ready',
            'database': 'not_initialized'
        }), 503

# =============================================================================
# API ROUTES - JOB MANAGEMENT
# =============================================================================

@app.route('/jobs', methods=['POST'])
def create_job_endpoint():
    """
    Create and start a new job
    
    Form parameters:
    - nb_an_projection: Number of years to project (default: 100)
    - nb_scenarios: Number of scenarios (default: 100)
    - max_accounts: Maximum number of accounts to process (optional)
    - debug_account: Account ID for debugging (optional)
    - use_custom_paths: Must be 'true' (default mode)
    
    Optional file paths (specify to override defaults):
    - population_path: Custom path for POPULATION.csv
    - mortalite_path: Custom path for MORTALITE.csv
    - rendements_path: Custom path for RENDEMENTS.csv
    - depots_futurs_path: Custom path for DEPOTS_FUTURS.csv
    - frais_admin_path: Custom path for FRAIS_ADMIN.csv
    - min_ferr_path: Custom path for MIN_FERR.csv
    - tx_lapse_part_path: Custom path for TX_LAPSE_PART.csv
    - tx_lapse_tot_path: Custom path for TX_LAPSE_TOT.csv
    - acquisition_path: Custom path for ACQUISITION.csv
    - coussins_escap_path: Custom path for COUSSINS_ESCAP.csv
    
    Note: Any unspecified paths will automatically use default files from data_in folder.
    """
    try:
        # Generate job ID
        job_id = f"job_{datetime.utcnow().strftime('%Y%m%d_%H%M%S_%f')}"
        
        # Get parameters from form
        parameters = {
            'nb_an_projection': int(request.form.get('nb_an_projection', 100)),
            'nb_scenarios': int(request.form.get('nb_scenarios', 100)),
            'max_accounts': int(request.form.get('max_accounts')) if request.form.get('max_accounts') else None,
            'debug_account': int(request.form.get('debug_account')) if request.form.get('debug_account') else None,
            'use_default_files': request.form.get('use_default_files') == 'true',
            'use_custom_paths': request.form.get('use_custom_paths') == 'true',
            # Custom file paths (optional)
            'population_path': request.form.get('population_path'),
            'mortalite_path': request.form.get('mortalite_path'),
            'rendements_path': request.form.get('rendements_path'),
            'depots_futurs_path': request.form.get('depots_futurs_path'),
            'frais_admin_path': request.form.get('frais_admin_path'),
            'min_ferr_path': request.form.get('min_ferr_path'),
            'tx_lapse_part_path': request.form.get('tx_lapse_part_path'),
            'tx_lapse_tot_path': request.form.get('tx_lapse_tot_path'),
            'acquisition_path': request.form.get('acquisition_path'),
            'coussins_escap_path': request.form.get('coussins_escap_path')
        }
        
        # Handle file uploads
        uploaded_files = []
        upload_folder = get_job_upload_folder(job_id)
        
        # Check for files uploaded as a batch (old method)
        if 'files' in request.files:
            files = request.files.getlist('files')
            for file in files:
                if file and file.filename and allowed_file(file.filename):
                    filename = secure_filename(file.filename)
                    filepath = upload_folder / filename
                    file.save(filepath)
                    uploaded_files.append(filename)
        
        # Check for individual file uploads with specific names
        expected_files = [
            'POPULATION.csv', 'MORTALITE.csv', 'RENDEMENTS.csv',
            'DEPOTS_FUTURS.csv', 'FRAIS_ADMIN.csv', 'MIN_FERR.csv',
            'TX_LAPSE_PART.csv', 'TX_LAPSE_TOT.csv', 'ACQUISITION.csv',
            'COUSSINS_ESCAP.csv'
        ]
        
        for expected_filename in expected_files:
            file_key = expected_filename.replace('.csv', '')
            if file_key in request.files:
                file = request.files[file_key]
                if file and file.filename and allowed_file(file.filename):
                    # Save with expected filename
                    filepath = upload_folder / expected_filename
                    file.save(filepath)
                    uploaded_files.append(expected_filename)
        
        # Check if we have either uploaded files, custom paths mode, or use default files
        use_custom_paths = parameters.get('use_custom_paths', False)
        use_default_files = parameters.get('use_default_files', False)
        
        # Allow job creation if we have files OR will use defaults
        # (We always use defaults for unspecified files now)
        if not uploaded_files and not use_custom_paths and not use_default_files:
            # Actually, it's OK to have no files - we'll use defaults
            pass
        
        # Create job in database
        create_job(job_id, parameters, uploaded_files)
        
        # Initialize cancellation flag
        job_cancellation_flags[job_id] = False
        
        # Start processing in background thread
        thread = threading.Thread(target=process_job, args=(job_id,))
        thread.daemon = True
        thread.start()
        
        # Track the thread
        job_threads[job_id] = thread
        
        return jsonify({
            'job_id': job_id,
            'status': 'pending',
            'message': 'Job created and started',
            'parameters': parameters,
            'uploaded_files': uploaded_files
        }), 201
        
    except Exception as e:
        return jsonify({
            'error': 'Failed to create job',
            'message': str(e)
        }), 500

@app.route('/jobs', methods=['GET'])
def list_jobs():
    """List all jobs sorted by creation date (newest first)"""
    try:
        jobs = get_all_jobs()
        return jsonify({
            'jobs': jobs,
            'count': len(jobs)
        })
    except Exception as e:
        return jsonify({
            'error': 'Failed to list jobs',
            'message': str(e)
        }), 500

@app.route('/jobs/<job_id>', methods=['GET'])
def get_job_details(job_id: str):
    """Get specific job details"""
    try:
        job = get_job(job_id)
        if not job:
            return jsonify({
                'error': 'Job not found',
                'job_id': job_id
            }), 404
        
        return jsonify(job)
        
    except Exception as e:
        return jsonify({
            'error': 'Failed to get job details',
            'message': str(e)
        }), 500

@app.route('/jobs/<job_id>', methods=['DELETE'])
def cancel_job(job_id: str):
    """Cancel a running or pending job"""
    try:
        job = get_job(job_id)
        if not job:
            return jsonify({
                'error': 'Job not found',
                'job_id': job_id
            }), 404
        
        # Check if job can be cancelled
        if job['status'] in ['completed', 'failed', 'cancelled']:
            return jsonify({
                'error': 'Job cannot be cancelled',
                'status': job['status'],
                'message': f'Job is already {job["status"]}'
            }), 400
        
        # Set cancellation flag
        job_cancellation_flags[job_id] = True
        
        # Update status to cancelled
        update_job_status(job_id, 'cancelled')
        
        return jsonify({
            'job_id': job_id,
            'status': 'cancelled',
            'message': 'Job cancellation requested'
        }), 200
        
    except Exception as e:
        return jsonify({
            'error': 'Failed to cancel job',
            'message': str(e)
        }), 500

# =============================================================================
# API ROUTES - RESULTS & FILES
# =============================================================================

@app.route('/jobs/<job_id>/results', methods=['GET'])
def get_job_results(job_id: str):
    """
    Get job results
    Query param type: 'summary' (default), 'detailed', or 'internal'
    Returns JSON data for the specified result type
    """
    try:
        # Get job
        job = get_job(job_id)
        if not job:
            return jsonify({
                'error': 'Job not found',
                'job_id': job_id
            }), 404
        
        # Check if job is completed
        if job['status'] != 'completed':
            return jsonify({
                'error': 'Job not completed',
                'status': job['status'],
                'message': 'Results are only available for completed jobs'
            }), 400
        
        # Get result type from query params
        result_type = request.args.get('type', 'summary').lower()
        
        # Map result types to files
        result_file_map = {
            'summary': 'VP_FLUX_TOTAL_GPU.csv',
            'detailed': 'VP_FLUX_COMPTE_GPU.csv',
            'internal': 'FLUX_PROJETES_GPU.csv'
        }
        
        if result_type not in result_file_map:
            return jsonify({
                'error': 'Invalid result type',
                'message': f'Type must be one of: {", ".join(result_file_map.keys())}'
            }), 400
        
        # Read the result file
        results_folder = get_job_results_folder(job_id)
        result_file = results_folder / result_file_map[result_type]
        
        if not result_file.exists():
            return jsonify({
                'error': 'Result file not found',
                'expected_file': result_file_map[result_type]
            }), 404
        
        # Read CSV and convert to JSON
        df = pd.read_csv(result_file, sep=';')
        data = df.to_dict(orient='records')
        
        return jsonify({
            'job_id': job_id,
            'result_type': result_type,
            'count': len(data),
            'data': data
        })
        
    except Exception as e:
        return jsonify({
            'error': 'Failed to get results',
            'message': str(e)
        }), 500

@app.route('/jobs/<job_id>/files', methods=['GET'])
def list_job_files(job_id: str):
    """List files uploaded with a job"""
    try:
        job = get_job(job_id)
        if not job:
            return jsonify({
                'error': 'Job not found',
                'job_id': job_id
            }), 404
        
        # Get uploaded files
        upload_folder = get_job_upload_folder(job_id)
        uploaded_files = []
        if upload_folder.exists():
            for file in upload_folder.glob('*'):
                if file.is_file():
                    uploaded_files.append({
                        'name': file.name,
                        'size': file.stat().st_size,
                        'type': 'input'
                    })
        
        # Get result files from database (with metadata)
        result_files = job.get('result_files', [])
        
        # If result_files are stored as strings (old format), convert to new format
        if result_files and isinstance(result_files[0], str):
            results_folder = get_job_results_folder(job_id)
            result_files = []
            for filename in job.get('result_files', []):
                file_path = results_folder / filename
                if file_path.exists():
                    result_files.append({
                        'name': filename,
                        'size': file_path.stat().st_size,
                        'type': 'output'
                    })
        
        return jsonify({
            'job_id': job_id,
            'uploaded_files': uploaded_files,
            'result_files': result_files,
            'total_files': len(uploaded_files) + len(result_files)
        })
        
    except Exception as e:
        return jsonify({
            'error': 'Failed to list files',
            'message': str(e)
        }), 500

@app.route('/jobs/<job_id>/files/<file_name>/preview', methods=['GET'])
def preview_file(job_id: str, file_name: str):
    """Preview a file with limited rows"""
    try:
        job = get_job(job_id)
        if not job:
            return jsonify({
                'error': 'Job not found',
                'job_id': job_id
            }), 404
        
        # Get row limit from query params (default 100)
        limit = int(request.args.get('limit', 100))
        
        # Secure the filename
        file_name = secure_filename(file_name)
        
        # Try to find file in results folder first
        results_folder = get_job_results_folder(job_id)
        file_path = results_folder / file_name
        
        if not file_path.exists():
            # Try upload folder
            upload_folder = get_job_upload_folder(job_id)
            file_path = upload_folder / file_name
        
        if not file_path.exists():
            return jsonify({
                'error': 'File not found',
                'file_name': file_name
            }), 404
        
        # Read CSV with limit
        df = pd.read_csv(file_path, sep=';', nrows=limit)
        
        # Get total row count
        total_rows = sum(1 for _ in open(file_path)) - 1  # Subtract header
        
        # Convert to JSON
        data = df.to_dict(orient='records')
        columns = df.columns.tolist()
        
        return jsonify({
            'job_id': job_id,
            'file_name': file_name,
            'columns': columns,
            'data': data,
            'rows_shown': len(data),
            'total_rows': total_rows,
            'truncated': total_rows > limit
        })
        
    except Exception as e:
        return jsonify({
            'error': 'Failed to preview file',
            'message': str(e)
        }), 500

@app.route('/jobs/<job_id>/files/<file_name>', methods=['GET'])
def download_file(job_id: str, file_name: str):
    """Download specific file content"""
    try:
        job = get_job(job_id)
        if not job:
            return jsonify({
                'error': 'Job not found',
                'job_id': job_id
            }), 404
        
        # Secure the filename
        file_name = secure_filename(file_name)
        
        # Check in upload folder first
        upload_file = get_job_upload_folder(job_id) / file_name
        if upload_file.exists():
            return send_file(upload_file, mimetype='text/plain', as_attachment=True)
        
        # Check in results folder
        result_file = get_job_results_folder(job_id) / file_name
        if result_file.exists():
            return send_file(result_file, mimetype='text/plain', as_attachment=True)
        
        return jsonify({
            'error': 'File not found',
            'file_name': file_name
        }), 404
        
    except Exception as e:
        return jsonify({
            'error': 'Failed to download file',
            'message': str(e)
        }), 500

# =============================================================================
# ERROR HANDLERS
# =============================================================================

@app.errorhandler(404)
def not_found(e):
    return jsonify({
        'error': 'Not found',
        'message': 'The requested resource was not found'
    }), 404

@app.errorhandler(500)
def internal_error(e):
    return jsonify({
        'error': 'Internal server error',
        'message': 'An unexpected error occurred'
    }), 500

# =============================================================================
# API ROUTES - ADMIN
# =============================================================================

@app.route('/admin/clear-database', methods=['POST'])
def clear_database():
    """Clear all jobs from database (requires password)"""
    try:
        # Get password from request
        password = request.json.get('password') if request.is_json else request.form.get('password')
        
        if not password:
            return jsonify({
                'error': 'Password required',
                'message': 'Please provide the admin password'
            }), 401
        
        # Verify password
        if password != ADMIN_PASSWORD:
            return jsonify({
                'error': 'Invalid password',
                'message': 'The provided password is incorrect'
            }), 403
        
        # Get options
        delete_files = request.json.get('delete_files', False) if request.is_json else request.form.get('delete_files') == 'true'
        
        # Clear database
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM jobs")
        deleted_count = cursor.rowcount
        conn.commit()
        conn.close()
        
        # Optionally delete uploaded and result files
        if delete_files:
            import shutil
            for folder in [app.config['UPLOAD_FOLDER'], app.config['RESULTS_FOLDER']]:
                if folder.exists():
                    for item in folder.iterdir():
                        if item.is_dir():
                            shutil.rmtree(item)
                        else:
                            item.unlink()
        
        # Clear job tracking
        job_threads.clear()
        job_cancellation_flags.clear()
        
        return jsonify({
            'success': True,
            'message': f'Database cleared successfully',
            'jobs_deleted': deleted_count,
            'files_deleted': delete_files
        }), 200
        
    except Exception as e:
        return jsonify({
            'error': 'Failed to clear database',
            'message': str(e)
        }), 500

# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    # Get port from environment (RunPod default is 80)
    port = int(os.getenv('PORT', 80))
    
    print("=" * 60)
    print("GPU Actuarial Projections API")
    print("=" * 60)
    print(f"Version: {APP_VERSION}")
    print(f"Environment: {ENVIRONMENT}")
    print(f"GPU Available: {GPU_AVAILABLE}")
    print(f"Database Type: {'NeonDB (PostgreSQL)' if USE_NEONDB else 'SQLite'}")
    if USE_NEONDB:
        print(f"Database URL: {NEONDB_URL.split('@')[1] if '@' in NEONDB_URL else 'configured'}")
    else:
        print(f"Database: {app.config['DATABASE']}")
    print(f"Upload Folder: {app.config['UPLOAD_FOLDER']}")
    print(f"Results Folder: {app.config['RESULTS_FOLDER']}")
    print(f"Admin Password: {'***' if ENVIRONMENT == 'production' else ADMIN_PASSWORD}")
    print(f"Port: {port}")
    print("=" * 60)
    if ENVIRONMENT != 'production':
        print("⚠️  WARNING: Using default admin password!")
        print("⚠️  Set ADMIN_PASSWORD environment variable for production")
        print("=" * 60)


    # Run the app
    app.run(
        host='0.0.0.0',
        port=port,
        debug=(ENVIRONMENT == 'development')
    )
