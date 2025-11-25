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
from typing import Optional, Dict, Any, Tuple
import pandas as pd
from contextlib import contextmanager

from flask import Flask, request, jsonify, send_file, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
import runpod
import requests

from paths import HERE

# Load environment variables from .env file
load_dotenv()

# Import PostgreSQL adapter if available
try:
    import psycopg
    from psycopg.rows import dict_row
    PSYCOPG_AVAILABLE = True
except ImportError:
    PSYCOPG_AVAILABLE = False
    print("Warning: psycopg not available, PostgreSQL support disabled")

# Import SQLAlchemy for pandas database operations
try:
    from sqlalchemy import create_engine
    SQLALCHEMY_AVAILABLE = True
except ImportError:
    SQLALCHEMY_AVAILABLE = False
    print("Warning: sqlalchemy not available, pandas database operations may fail")

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

# CORS Configuration
# Disable Flask-CORS when RUNPOD_CORS=true (RunPod handles CORS automatically)
# This prevents duplicate Access-Control-Allow-Origin headers
ENABLE_FLASK_CORS = os.getenv('RUNPOD_CORS', 'false').lower() != 'true'

if ENABLE_FLASK_CORS:
    # Enable CORS for local development or non-RunPod deployments
    CORS(app, resources={
        r"/*": {
            "origins": os.getenv('CORS_ORIGINS', '*').split(','),
            "methods": ["GET", "POST", "DELETE", "OPTIONS"],
            "allow_headers": ["Content-Type", "Authorization"]
        }
    })
    print("✓ Flask-CORS enabled")
else:
    print("✓ Flask-CORS disabled (RunPod handles CORS)")

app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max upload
app.config['UPLOAD_FOLDER'] = HERE / 'uploads'
app.config['RESULTS_FOLDER'] = HERE / 'results'
app.config['DATABASE'] = HERE / 'jobs.db'
app.config['DEFAULT_DATA_FOLDER'] = HERE/ 'data_in'

# Create directories if they don't exist
app.config['UPLOAD_FOLDER'].mkdir(exist_ok=True)
app.config['RESULTS_FOLDER'].mkdir(exist_ok=True)
HERE.joinpath('static').mkdir(exist_ok=True)

# Application metadata
APP_VERSION = "1.0.0"
ENVIRONMENT = os.getenv('ENVIRONMENT', 'development')
ADMIN_PASSWORD = os.getenv('ADMIN_PASSWORD', 'admin123')  # Change in production!

# RunPod worker configuration
# NOTE: Use QUEUE-BASED endpoint, not Load Balancing endpoint
RUNPOD_API_KEY = os.getenv('RUNPOD_API_KEY')
RUNPOD_ENDPOINT_ID = os.getenv('RUNPOD_ENDPOINT_ID')

# RunPod configuration
PORT = int(os.getenv('PORT', '80'))  # Main application port
PORT_HEALTH = int(os.getenv('PORT_HEALTH', str(PORT)))  # Health check port (defaults to PORT)

# Allowed file extensions
ALLOWED_EXTENSIONS = {'csv'}

# Database configuration
USE_NEONDB = os.getenv('USE_NEONDB', 'false').lower() == 'true'
NEONDB_URL = os.getenv('NEONDB_URL', '')

# Determine which database to use
if USE_NEONDB and PSYCOPG_AVAILABLE and NEONDB_URL:
    DATABASE_TYPE = 'postgresql'
    print(f"Using PostgreSQL/NeonDB: {NEONDB_URL.split('@')[1].split('/')[0] if '@' in NEONDB_URL else 'configured'}")
else:
    DATABASE_TYPE = 'sqlite'
    if USE_NEONDB and not PSYCOPG_AVAILABLE:
        print("Warning: USE_NEONDB=true but psycopg not available, falling back to SQLite")
    elif USE_NEONDB and not NEONDB_URL:
        print("Warning: USE_NEONDB=true but NEONDB_URL not set, falling back to SQLite")

# =============================================================================
# DATABASE ABSTRACTION LAYER
# =============================================================================

@contextmanager
def get_db_cursor():
    """Get a database cursor with automatic connection management"""
    if DATABASE_TYPE == 'postgresql':
        conn = psycopg.connect(NEONDB_URL, row_factory=dict_row)
        try:
            cursor = conn.cursor()
            yield cursor, conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            cursor.close()
            conn.close()
    else:
        conn = sqlite3.connect(app.config['DATABASE'])
        conn.row_factory = sqlite3.Row
        try:
            cursor = conn.cursor()
            yield cursor, conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            cursor.close()
            conn.close()

def execute_sql(sql: str, params: Tuple = ()) -> None:
    """Execute SQL with parameters"""
    with get_db_cursor() as (cursor, conn):
        cursor.execute(sql, params)

def fetch_one(sql: str, params: Tuple = ()) -> Optional[Dict]:
    """Fetch one row as dictionary"""
    with get_db_cursor() as (cursor, conn):
        cursor.execute(sql, params)
        row = cursor.fetchone()
        if row:
            return dict(row) if DATABASE_TYPE == 'postgresql' else {k: row[k] for k in row.keys()}
        return None

def fetch_all(sql: str, params: Tuple = ()) -> list:
    """Fetch all rows as list of dictionaries"""
    with get_db_cursor() as (cursor, conn):
        cursor.execute(sql, params)
        rows = cursor.fetchall()
        if DATABASE_TYPE == 'postgresql':
            return [dict(row) for row in rows]
        else:
            return [{k: row[k] for k in row.keys()} for row in rows]

def get_placeholder():
    """Get the parameter placeholder for the current database type"""
    return '%s' if DATABASE_TYPE == 'postgresql' else '?'

# =============================================================================
# DATABASE INITIALIZATION
# =============================================================================

def init_db():
    """Initialize the database with required tables (supports both SQLite and PostgreSQL)"""
    
    # Determine ID column based on database type
    id_column = "SERIAL PRIMARY KEY" if DATABASE_TYPE == 'postgresql' else "INTEGER PRIMARY KEY AUTOINCREMENT"
    
    jobs_table = f"""
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
            progress_percent REAL DEFAULT 0.0,
            results_data TEXT
        )
    """
    
    flux_projetes_table = f"""
        CREATE TABLE IF NOT EXISTS flux_projetes (
            id {id_column},
            job_id TEXT NOT NULL,
            an_eval INTEGER NOT NULL,
            mois_eval INTEGER NOT NULL,
            primes_garanties REAL,
            prest_deces REAL,
            prest_ech REAL,
            prest_mrv REAL,
            frais_acquis REAL,
            comm_vente REAL,
            primes_variables REAL,
            frais_fixes REAL,
            hon_gest REAL,
            comm_maintien REAL,
            valeur_marchande REAL,
            passif_redresse REAL,
            coussin_credit REAL,
            coussin_marche REAL,
            coussin_depense REAL,
            coussin_decheance REAL,
            coussin_mortalite REAL,
            coussin_depot REAL,
            FOREIGN KEY (job_id) REFERENCES jobs(job_id) ON DELETE CASCADE
        )
    """
    
    vp_flux_compte_table = f"""
        CREATE TABLE IF NOT EXISTS vp_flux_compte (
            id {id_column},
            job_id TEXT NOT NULL,
            id_compte INTEGER NOT NULL,
            vp_frais_acquis REAL,
            vp_comm_vente REAL,
            vp_primes_garanties REAL,
            vp_primes_variables REAL,
            vp_frais_fixes REAL,
            vp_hon_gest REAL,
            vp_comm_maintien REAL,
            vp_prest_ech REAL,
            vp_prest_mrv REAL,
            vp_prest_deces REAL,
            vp_passif_redresse REAL,
            vp_coussin_credit REAL,
            vp_coussin_marche REAL,
            vp_coussin_depense REAL,
            vp_coussin_decheance REAL,
            vp_coussin_mortalite REAL,
            vp_coussin_depot REAL,
            vp_valeur_marchande REAL,
            FOREIGN KEY (job_id) REFERENCES jobs(job_id) ON DELETE CASCADE
        )
    """
    
    vp_flux_total_table = f"""
        CREATE TABLE IF NOT EXISTS vp_flux_total (
            id {id_column},
            job_id TEXT NOT NULL,
            categorie TEXT NOT NULL,
            vp_flux_tot REAL NOT NULL,
            FOREIGN KEY (job_id) REFERENCES jobs(job_id) ON DELETE CASCADE
        )
    """
    
    with get_db_cursor() as (cursor, conn):
        # Create tables
        cursor.execute(jobs_table)
        cursor.execute(flux_projetes_table)
        cursor.execute(vp_flux_compte_table)
        cursor.execute(vp_flux_total_table)
        
        # Create indexes
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_flux_projetes_job_id 
            ON flux_projetes(job_id)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_vp_flux_compte_job_id 
            ON vp_flux_compte(job_id)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_vp_flux_total_job_id 
            ON vp_flux_total(job_id)
        """)
        
        # Handle migrations for SQLite only (PostgreSQL schema has all columns from start)
        if DATABASE_TYPE == 'sqlite':
            try:
                cursor.execute("SELECT current_batch FROM jobs LIMIT 1")
            except sqlite3.OperationalError:
                print("Migrating database: Adding progress tracking columns...")
                cursor.execute("ALTER TABLE jobs ADD COLUMN current_batch INTEGER DEFAULT 0")
                cursor.execute("ALTER TABLE jobs ADD COLUMN total_batches INTEGER DEFAULT 0")
                cursor.execute("ALTER TABLE jobs ADD COLUMN progress_percent REAL DEFAULT 0.0")
                print("Migration complete")
            
            try:
                cursor.execute("SELECT results_data FROM jobs LIMIT 1")
            except sqlite3.OperationalError:
                print("Migrating database: Adding results_data column...")
                cursor.execute("ALTER TABLE jobs ADD COLUMN results_data TEXT")
                print("Migration complete")

# Initialize database on startup
init_db()

# Application health state for RunPod load balancing
# States: 'initializing' (204), 'healthy' (200), 'unhealthy' (503)
app_health_state = 'initializing'
db_initialized = False

def set_app_healthy():
    """Mark the application as healthy and ready to serve requests"""
    global app_health_state, db_initialized
    app_health_state = 'healthy'
    db_initialized = True
    print("✓ Application is healthy and ready to serve requests")

# Set app as healthy after database initialization
set_app_healthy()

# Track running jobs and their threads
job_threads = {}
job_cancellation_flags = {}
if RUNPOD_API_KEY:
    runpod.api_key = RUNPOD_API_KEY
    print("✓ RunPod API key configured")
else:
    print("Warning: RUNPOD_API_KEY not set. RunPod worker integration will be disabled.")

job_progress = {}  # In-memory progress tracking: {job_id: {'current': int, 'total': int}}

# =============================================================================
# DATABASE HELPERS
# =============================================================================

def get_db_connection():
    """Get a database connection (for backward compatibility with pandas.to_sql)"""
    if DATABASE_TYPE == 'postgresql':
        return psycopg.connect(NEONDB_URL)
    else:
        conn = sqlite3.connect(app.config['DATABASE'])
        conn.row_factory = sqlite3.Row
        return conn

def get_sqlalchemy_engine():
    """Get a SQLAlchemy engine for pandas operations"""
    if DATABASE_TYPE == 'postgresql':
        # Convert psycopg URL to SQLAlchemy-compatible URL
        # psycopg uses postgresql:// but we need to ensure proper format
        if NEONDB_URL.startswith('postgresql://'):
            url = NEONDB_URL.replace('postgresql://', 'postgresql+psycopg://', 1)
        else:
            url = NEONDB_URL
        return create_engine(url)
    else:
        return create_engine(f'sqlite:///{app.config["DATABASE"]}')

def create_job(job_id: str, parameters: Dict[str, Any], uploaded_files: list) -> None:
    """Create a new job in the database"""
    ph = get_placeholder()
    sql = f"""
        INSERT INTO jobs (job_id, status, created_at, parameters, uploaded_files)
        VALUES ({ph}, {ph}, {ph}, {ph}, {ph})
    """
    execute_sql(sql, (
        job_id,
        'pending',
        datetime.utcnow().isoformat(),
        json.dumps(parameters),
        json.dumps(uploaded_files)
    ))

def update_job_status(job_id: str, status: str, error_message: Optional[str] = None, progress_message: Optional[str] = None) -> None:
    """Update job status and optionally set progress/error message"""
    ph = get_placeholder()
    updates = {'status': status}
    
    if status == 'running' and error_message is None and progress_message is None:
        updates['started_at'] = datetime.utcnow().isoformat()
    elif status in ['completed', 'failed']:
        updates['completed_at'] = datetime.utcnow().isoformat()
    
    # Use progress_message for running status, error_message for failures
    if progress_message and status == 'running':
        updates['error_message'] = progress_message  # Reuse error_message field for progress
    elif error_message:
        updates['error_message'] = error_message
    
    set_clause = ', '.join([f"{k} = {ph}" for k in updates.keys()])
    values = list(updates.values()) + [job_id]
    
    sql = f"UPDATE jobs SET {set_clause} WHERE job_id = {ph}"
    execute_sql(sql, tuple(values))

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
    ph = get_placeholder()
    sql = f"""
        UPDATE jobs 
        SET current_batch = {ph}, total_batches = {ph}, progress_percent = {ph}
        WHERE job_id = {ph}
    """
    execute_sql(sql, (current_batch, total_batches, progress_percent, job_id))

def update_job_results(job_id: str, result_files: list) -> None:
    """
    Update job with result files
    
    Args:
        job_id: Job identifier
        result_files: List of file dictionaries with keys: name, type, description, size
    """
    ph = get_placeholder()
    sql = f"UPDATE jobs SET result_files = {ph} WHERE job_id = {ph}"
    execute_sql(sql, (json.dumps(result_files), job_id))

def update_job_results_data(job_id: str, results_data: dict) -> None:
    """
    Update job with results data from run_projection_gpu (legacy JSON storage)
    
    Args:
        job_id: Job identifier
        results_data: Dictionary containing results from run_projection_gpu
    """
    ph = get_placeholder()
    sql = f"UPDATE jobs SET results_data = {ph} WHERE job_id = {ph}"
    execute_sql(sql, (json.dumps(results_data), job_id))

def save_flux_projetes(job_id: str, df: pd.DataFrame) -> None:
    """Save flux_projetes DataFrame to database table"""
    engine = get_sqlalchemy_engine()
    
    # Prepare data for bulk insert
    df_copy = df.copy()
    df_copy.insert(0, 'job_id', job_id)
    
    # Rename columns to match database schema (lowercase)
    df_copy.columns = df_copy.columns.str.lower()
    
    # Insert into database
    df_copy.to_sql('flux_projetes', engine, if_exists='append', index=False)
    engine.dispose()
    print(f"  Saved {len(df)} flux_projetes records to database")

def save_vp_flux_compte(job_id: str, df: pd.DataFrame) -> None:
    """Save vp_flux_compte DataFrame to database table"""
    engine = get_sqlalchemy_engine()
    
    # Prepare data for bulk insert
    df_copy = df.copy()
    df_copy.insert(0, 'job_id', job_id)
    
    # Rename columns to match database schema (lowercase)
    df_copy.columns = df_copy.columns.str.lower()
    
    # Insert into database
    df_copy.to_sql('vp_flux_compte', engine, if_exists='append', index=False)
    engine.dispose()
    print(f"  Saved {len(df)} vp_flux_compte records to database")

def save_vp_flux_total(job_id: str, df: pd.DataFrame) -> None:
    """Save vp_flux_total DataFrame to database table"""
    engine = get_sqlalchemy_engine()
    
    # Prepare data for bulk insert
    df_copy = df.copy()
    df_copy.insert(0, 'job_id', job_id)
    
    # Rename columns to match database schema (lowercase)
    df_copy.columns = df_copy.columns.str.lower()
    
    # Insert into database
    df_copy.to_sql('vp_flux_total', engine, if_exists='append', index=False)
    engine.dispose()
    print(f"  Saved {len(df)} vp_flux_total records to database")

def get_flux_projetes(job_id: str, an_eval: int = None, mois_eval: int = None) -> Optional[pd.DataFrame]:
    """
    Retrieve flux_projetes results for a job with optional filters
    
    Args:
        job_id: Job identifier
        an_eval: Filter by year (optional)
        mois_eval: Filter by month (optional)
    """
    conn = get_db_connection()
    try:
        # Build query with filters
        ph = get_placeholder()
        query = f"SELECT * FROM flux_projetes WHERE job_id = {ph}"
        params = [job_id]
        
        if an_eval is not None:
            query += f" AND an_eval = {ph}"
            params.append(an_eval)
        
        if mois_eval is not None:
            query += f" AND mois_eval = {ph}"
            params.append(mois_eval)
        
        query += " ORDER BY an_eval, mois_eval"
        
        df = pd.read_sql_query(query, conn, params=params)
        conn.close()
        if len(df) > 0:
            # Remove database-specific columns
            df = df.drop(columns=['id', 'job_id'])
            # Restore column names to uppercase
            df.columns = df.columns.str.upper()
            return df
        return None
    except Exception as e:
        conn.close()
        print(f"Error retrieving flux_projetes: {e}")
        return None

def get_vp_flux_compte(job_id: str, id_compte: int = None) -> Optional[pd.DataFrame]:
    """
    Retrieve vp_flux_compte results for a job with optional filter
    
    Args:
        job_id: Job identifier
        id_compte: Filter by account ID (optional)
    """
    conn = get_db_connection()
    try:
        # Build query with filter
        ph = get_placeholder()
        query = f"SELECT * FROM vp_flux_compte WHERE job_id = {ph}"
        params = [job_id]
        
        if id_compte is not None:
            query += f" AND id_compte = {ph}"
            params.append(id_compte)
        
        query += " ORDER BY id_compte"
        
        df = pd.read_sql_query(query, conn, params=params)
        conn.close()
        if len(df) > 0:
            # Remove database-specific columns
            df = df.drop(columns=['id', 'job_id'])
            # Restore column names to uppercase
            df.columns = df.columns.str.upper()
            return df
        return None
    except Exception as e:
        conn.close()
        print(f"Error retrieving vp_flux_compte: {e}")
        return None

def get_vp_flux_total(job_id: str) -> Optional[pd.DataFrame]:
    """Retrieve vp_flux_total results for a job"""
    conn = get_db_connection()
    try:
        ph = get_placeholder()
        df = pd.read_sql_query(
            f"SELECT * FROM vp_flux_total WHERE job_id = {ph}",
            conn,
            params=(job_id,)
        )
        conn.close()
        if len(df) > 0:
            # Remove database-specific columns
            df = df.drop(columns=['id', 'job_id'])
            # Restore column names to uppercase
            df.columns = df.columns.str.upper()
            return df
        return None
    except Exception as e:
        conn.close()
        print(f"Error retrieving vp_flux_total: {e}")
        return None

def get_job(job_id: str) -> Optional[Dict[str, Any]]:
    """Get job details by ID"""
    ph = get_placeholder()
    sql = f"SELECT * FROM jobs WHERE job_id = {ph}"
    row = fetch_one(sql, (job_id,))
    
    if row:
        job_data = {
            'job_id': row['job_id'],
            'status': row['status'],
            'created_at': row['created_at'],
            'started_at': row['started_at'],
            'completed_at': row['completed_at'],
            'error_message': row['error_message'],
            'parameters': json.loads(row['parameters']) if row['parameters'] else {},
            'uploaded_files': json.loads(row['uploaded_files']) if row['uploaded_files'] else [],
            'result_files': json.loads(row['result_files']) if row['result_files'] else []
        }
        
        # Handle progress fields (may not exist in old records)
        try:
            job_data['current_batch'] = row['current_batch'] if row['current_batch'] is not None else 0
            job_data['total_batches'] = row['total_batches'] if row['total_batches'] is not None else 0
            job_data['progress_percent'] = row['progress_percent'] if row['progress_percent'] is not None else 0.0
        except (KeyError, IndexError):
            job_data['current_batch'] = 0
            job_data['total_batches'] = 0
            job_data['progress_percent'] = 0.0
        
        # Handle results_data field (may not exist in old records)
        try:
            job_data['results_data'] = json.loads(row['results_data']) if row['results_data'] else None
        except (KeyError, IndexError):
            job_data['results_data'] = None
        
        return job_data
    return None

def poll_runpod_results(job_id: str, run_request):
    """
    Poll RunPod job for completion and save results to database.
    This runs in a background thread.
    """
    import time
    
    try:
        print(f"Starting result polling for job {job_id}")
        
        # Poll for job completion (timeout after 30 minutes)
        max_wait_time = 30 * 60  # 30 minutes
        poll_interval = 5  # Check every 5 seconds
        elapsed_time = 0
        
        while elapsed_time < max_wait_time:
            try:
                # Check job status
                status = run_request.status()
                print(f"  RunPod job status: {status}")
                
                # Update progress in database based on status
                if status == "IN_QUEUE":
                    update_job_progress(job_id, 0, 1, 0)
                    update_job_status(job_id, 'running', progress_message="⏳ Job queued on RunPod, waiting for GPU worker...")
                elif status == "IN_PROGRESS":
                    # Get job parameters to estimate progress
                    job = get_job(job_id)
                    params = job.get('parameters', {})
                    nb_scenarios = params.get('nb_scenarios', 100)
                    max_accounts = params.get('max_accounts')
                    
                    # Estimate time based on typical performance
                    # Rough estimate: ~30 seconds per 20k accounts with 100 scenarios
                    # Or ~1 second per 1k accounts with 10 scenarios
                    if max_accounts and nb_scenarios:
                        # Estimate total time (very rough)
                        estimated_seconds = (max_accounts / 1000) * (nb_scenarios / 10)
                    else:
                        # Assume full dataset (~200k accounts)
                        estimated_seconds = (200000 / 1000) * (nb_scenarios / 10) if nb_scenarios else 300
                    
                    # Calculate estimated progress
                    if estimated_seconds > 0 and elapsed_time > 0:
                        progress_pct = min(int((elapsed_time / estimated_seconds) * 100), 99)
                    else:
                        progress_pct = min(50 + (elapsed_time / max_wait_time * 50), 99)
                    
                    # Build informative message
                    minutes = int(elapsed_time / 60)
                    seconds = int(elapsed_time % 60)
                    msg = f"🚀 GPU projection in progress... ({minutes}m {seconds}s elapsed"
                    if estimated_seconds > 60:
                        est_total_min = int(estimated_seconds / 60)
                        msg += f", est. ~{est_total_min}m total"
                    msg += ")"
                    
                    update_job_progress(job_id, 1, 1, int(progress_pct))
                    update_job_status(job_id, 'running', progress_message=msg)
                
                if status == "COMPLETED":
                    # Get the output
                    output = run_request.output()
                    print(f"  RunPod job completed! Processing results...")
                    
                    # Check if output contains results
                    if output and isinstance(output, dict):
                        if 'results' in output:
                            # Convert results to DataFrame format and save to database
                            results_data = output['results']
                            print(f"  Processing results from RunPod worker...")
                            
                            # Save legacy JSON format
                            update_job_results_data(job_id, output)
                            
                            # Convert JSON results back to DataFrames and save to proper tables
                            saved_any = False
                            if isinstance(results_data, dict):
                                # Save flux_projetes
                                if results_data.get('flux_projetes'):
                                    df = pd.DataFrame(results_data['flux_projetes'])
                                    save_flux_projetes(job_id, df)
                                    saved_any = True
                                
                                # Save vp_flux_compte
                                if results_data.get('vp_flux_compte'):
                                    df = pd.DataFrame(results_data['vp_flux_compte'])
                                    save_vp_flux_compte(job_id, df)
                                    saved_any = True
                                
                                # Save vp_flux_total
                                if results_data.get('vp_flux_total'):
                                    df = pd.DataFrame(results_data['vp_flux_total'])
                                    save_vp_flux_total(job_id, df)
                                    saved_any = True
                                    print(f"  ✓ Total PV: ${df['VP_FLUX_TOT'].iloc[0]:,.2f}")
                            
                            if saved_any:
                                print(f"✓ Job {job_id} completed and results saved to database!")
                            else:
                                print(f"✓ Job {job_id} completed (no DataFrame results to save)")
                            
                            # Clear progress message and mark as completed
                            ph = get_placeholder()
                            sql = f"UPDATE jobs SET status = {ph}, completed_at = {ph}, error_message = NULL WHERE job_id = {ph}"
                            execute_sql(sql, ('completed', datetime.utcnow().isoformat(), job_id))
                            return
                        elif 'error' in output:
                            error_msg = output.get('error', 'Unknown error from worker')
                            if 'traceback' in output:
                                error_msg += f"\n\nWorker Traceback:\n{output['traceback']}"
                            update_job_status(job_id, 'failed', error_message=error_msg)
                            print(f"✗ Job {job_id} failed on worker: {error_msg}")
                            return
                    
                    # No recognizable output format
                    ph = get_placeholder()
                    sql = f"UPDATE jobs SET status = {ph}, completed_at = {ph}, error_message = NULL WHERE job_id = {ph}"
                    execute_sql(sql, ('completed', datetime.utcnow().isoformat(), job_id))
                    print(f"✓ Job {job_id} completed (no results data)")
                    return
                    
                elif status == "FAILED":
                    error_msg = "RunPod job failed"
                    try:
                        output = run_request.output()
                        if output and isinstance(output, dict) and 'error' in output:
                            error_msg = output['error']
                    except:
                        pass
                    update_job_status(job_id, 'failed', error_message=error_msg)
                    print(f"✗ Job {job_id} failed: {error_msg}")
                    return
                
                # Job still running, wait and check again
                time.sleep(poll_interval)
                elapsed_time += poll_interval
                
            except Exception as poll_error:
                print(f"  Error polling job status: {poll_error}")
                time.sleep(poll_interval)
                elapsed_time += poll_interval
        
        # Timeout reached
        error_msg = f"Job timed out after {max_wait_time/60} minutes"
        update_job_status(job_id, 'failed', error_message=error_msg)
        print(f"✗ Job {job_id} timed out")
        
    except Exception as e:
        error_msg = f"Error polling RunPod results: {str(e)}\n{traceback.format_exc()}"
        print(f"✗ Polling error for job {job_id}: {error_msg}")
        update_job_status(job_id, 'failed', error_message=error_msg)

def trigger_runpod_job(job_id: str):
    """
    Triggers a job on a RunPod serverless worker.
    Sends uploaded CSVs (or defaults) to the worker as base64-encoded data.
    """
    try:
        update_job_status(job_id, 'running')
        job = get_job(job_id)
        if not job:
            raise Exception(f"Job {job_id} not found")

        params = job['parameters']
        default_data_path = app.config['DEFAULT_DATA_FOLDER']
        upload_folder = get_job_upload_folder(job_id)
        uploaded_files = job.get('uploaded_files', [])

        # --- Prepare data files ---
        required_files = [
            'POPULATION.csv', 'MORTALITE.csv', 'RENDEMENTS.csv', 'DEPOTS_FUTURS.csv',
            'FRAIS_ADMIN.csv', 'MIN_FERR.csv', 'TX_LAPSE_PART.csv', 'TX_LAPSE_TOT.csv',
            'ACQUISITION.csv', 'COUSSINS_ESCAP.csv'
        ]

        print(f"Preparing CSV files for RunPod worker (job {job_id})...")
        # Upload files to tmpfiles.org and get URLs
        data_file_urls = {}
        
        if uploaded_files:
            print("  Uploading files to temporary hosting...")
            for filename in uploaded_files:
                if filename in required_files:
                    file_path = upload_folder / filename
                    
                    if file_path.exists():
                        try:
                            # Upload to tmpfiles.org
                            with open(file_path, 'rb') as f:
                                files = {'file': (filename, f, 'text/csv')}
                                response = requests.post(
                                    'https://tmpfiles.org/api/v1/upload',
                                    files=files,
                                    timeout=60
                                )
                                response.raise_for_status()
                                result = response.json()
                                
                                # Extract URL from response
                                if result.get('status') == 'success' and result.get('data', {}).get('url'):
                                    temp_url = result['data']['url']
                                    # Convert tmpfiles.org URL to direct download URL
                                    # https://tmpfiles.org/12345/file.csv -> https://tmpfiles.org/dl/12345/file.csv
                                    download_url = temp_url.replace('tmpfiles.org/', 'tmpfiles.org/dl/')
                                    data_file_urls[filename] = download_url
                                    print(f"  ✓ {filename}: uploaded ({file_path.stat().st_size} bytes)")
                                else:
                                    raise Exception(f"Upload failed: {result}")
                        except Exception as e:
                            raise Exception(f"Failed to upload {filename}: {str(e)}")
                    else:
                        raise FileNotFoundError(f"Uploaded file not found: {filename}")
            print(f"  Successfully uploaded {len(data_file_urls)} files")
        else:
            print("  Using all default CSVs from worker image (no uploads)")

        # --- Trigger RunPod job ---
        # Note: endpoint.run() automatically wraps input in {'input': ...}
        runpod_input = {
            'nb_an_projection': params.get('nb_an_projection', 10),
            'nb_scenarios': params.get('nb_scenarios', 100),
            'data_file_urls': data_file_urls  # Send URLs instead of file data
        }
        
        # Add optional parameters if provided
        if params.get('max_accounts'):
            runpod_input['max_accounts'] = params.get('max_accounts')
        if params.get('debug_account'):
            runpod_input['debug_account'] = params.get('debug_account')
        if params.get('debug_scenario'):
            runpod_input['debug_scenario'] = params.get('debug_scenario')

        print(f"Triggering RunPod job for endpoint {RUNPOD_ENDPOINT_ID}...")
        print(f"  Payload: {len(data_file_urls)} file URLs")
        print(f"  Input data: {json.dumps(runpod_input, indent=2)}")
        
        endpoint = runpod.Endpoint(RUNPOD_ENDPOINT_ID)
        
        # Check endpoint health first
        try:
            health = endpoint.health()
            print(f"  Endpoint health: {health}")
        except Exception as health_err:
            print(f"  Warning: Could not check endpoint health: {health_err}")
        
        # Trigger the RunPod job asynchronously (non-blocking)
        # Per docs: endpoint.run() returns immediately with a job request object
        run_request = endpoint.run(runpod_input)
        
        # Get the RunPod job ID (the SDK returns an object with .job_id attribute)
        runpod_job_id = run_request.job_id
        print(f"  RunPod job queued with ID: {runpod_job_id}")

        # Store the RunPod job ID for tracking
        ph = get_placeholder()
        sql = f"UPDATE jobs SET parameters = {ph} WHERE job_id = {ph}"
        params['runpod_job_id'] = runpod_job_id
        execute_sql(sql, (json.dumps(params), job_id))

        print(f"✓ RunPod job successfully queued!")
        print(f"  RunPod Job ID: {runpod_job_id}")
        print(f"  Local Job ID: {job_id}")
        print(f"  Status endpoint: /jobs/{job_id}")
        
        # Start polling for results in background
        poll_thread = threading.Thread(target=poll_runpod_results, args=(job_id, run_request))
        poll_thread.daemon = True
        poll_thread.start()

    except requests.exceptions.HTTPError as e:
        error_msg = f"{str(e)}\n{traceback.format_exc()}"
        
        # Extract details from HTTP error response
        if e.response is not None:
            try:
                error_details = e.response.json()
                error_msg += f"\n\nRunPod API Response: {json.dumps(error_details, indent=2)}"
            except:
                error_msg += f"\n\nRunPod API Response Text: {e.response.text}"
                error_msg += f"\n\nRunPod API Status Code: {e.response.status_code}"
        
        print(f"RunPod job trigger for {job_id} failed: {error_msg}")
        update_job_status(job_id, 'failed', error_message=error_msg)
    except Exception as e:
        error_msg = f"{str(e)}\n{traceback.format_exc()}"
        print(f"RunPod job trigger for {job_id} failed: {error_msg}")
        update_job_status(job_id, 'failed', error_message=error_msg)

def get_all_jobs() -> list:
    """Get all jobs ordered by creation date"""
    sql = "SELECT * FROM jobs ORDER BY created_at DESC"
    rows = fetch_all(sql)
    
    jobs = []
    for row in rows:
        job_data = {
            'job_id': row['job_id'],
            'status': row['status'],
            'created_at': row['created_at'],
            'started_at': row['started_at'],
            'completed_at': row['completed_at'],
            'error_message': row['error_message'],
            'parameters': json.loads(row['parameters']) if row['parameters'] else {},
            'uploaded_files': json.loads(row['uploaded_files']) if row['uploaded_files'] else [],
            'result_files': json.loads(row['result_files']) if row['result_files'] else []
        }
        
        # Handle progress fields (may not exist in old records)
        try:
            job_data['current_batch'] = row['current_batch'] if row['current_batch'] is not None else 0
            job_data['total_batches'] = row['total_batches'] if row['total_batches'] is not None else 0
            job_data['progress_percent'] = row['progress_percent'] if row['progress_percent'] is not None else 0.0
        except (KeyError, IndexError):
            job_data['current_batch'] = 0
            job_data['total_batches'] = 0
            job_data['progress_percent'] = 0.0
        
        # Handle results_data field (may not exist in old records)
        try:
            job_data['results_data'] = json.loads(row['results_data']) if row['results_data'] else None
        except (KeyError, IndexError):
            job_data['results_data'] = None
        
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
    
    NOTE: This function runs jobs locally with GPU. 
    If RunPod is configured, jobs should use trigger_runpod_job() instead.
    """
    try:
        # Prevent local execution if RunPod is configured
        if RUNPOD_ENDPOINT_ID and RUNPOD_API_KEY:
            error_msg = (
                "This job was created for local execution, but RunPod is now configured. "
                "Local GPU execution is disabled when RunPod is available. "
                "Please create a new job which will automatically use RunPod workers."
            )
            print(f"ERROR: {error_msg}")
            update_job_status(job_id, 'failed', error_message=error_msg)
            return
        
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
        
        # Save results data to separate database tables
        if results:
            print(f"\nSaving results to database...")
            if 'flux_projetes' in results and results['flux_projetes'] is not None:
                save_flux_projetes(job_id, results['flux_projetes'])
            if 'vp_flux_compte' in results and results['vp_flux_compte'] is not None:
                save_vp_flux_compte(job_id, results['vp_flux_compte'])
            if 'vp_flux_total' in results and results['vp_flux_total'] is not None:
                save_vp_flux_total(job_id, results['vp_flux_total'])
            
            # Also save summary to JSON for backward compatibility
            results_data = {
                'vp_flux_total_summary': {
                    'vp_flux_tot': float(results['vp_flux_total']['VP_FLUX_TOT'].iloc[0]) if 'vp_flux_total' in results and len(results['vp_flux_total']) > 0 else 0.0
                },
                'vp_flux_compte_summary': {
                    'total_accounts': len(results['vp_flux_compte']) if 'vp_flux_compte' in results else 0
                },
                'flux_projetes_summary': {
                    'total_periods': len(results['flux_projetes']) if 'flux_projetes' in results else 0
                }
            }
            update_job_results_data(job_id, results_data)
        
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

# =============================================================================
# API ENDPOINTS
# =============================================================================

@app.route('/jobs/runpod', methods=['POST'])
def create_runpod_job_endpoint():
    """
    Endpoint to create and trigger a new projection job on a RunPod worker.
    """
    if not RUNPOD_ENDPOINT_ID or not RUNPOD_API_KEY:
        return jsonify({'error': 'RunPod environment variables (RUNPOD_ENDPOINT_ID, RUNPOD_API_KEY) are not configured.'}), 500

    job_id = f"job_{datetime.utcnow().strftime('%Y%m%d%H%M%S%f')}"
    params = request.json or {}

    # Create job record in local DB
    create_job(job_id, params, uploaded_files=[])

    # Trigger the RunPod job in a background thread
    thread = threading.Thread(target=trigger_runpod_job, args=(job_id,))
    thread.start()
    job_threads[job_id] = thread

    return jsonify({'job_id': job_id, 'status': 'pending'}), 202

# =============================================================================
# API ROUTES - HEALTH & STATUS
# =============================================================================

@app.route('/', methods=['GET'])
def serve_index():
    """Serve the index.html file from the static folder."""
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/info', methods=['GET'])
def welcome():
    """Welcome endpoint with API information"""
    return jsonify({
        'message': 'GPU Actuarial Projections API',
        'version': APP_VERSION,
        'environment': ENVIRONMENT,
        'gpu_available': GPU_AVAILABLE,
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
    """
    Health check endpoint for RunPod load balancing
    Returns:
        200: Application is healthy and ready
        204: Application is initializing
        503: Application is unhealthy
    """
    if app_health_state == 'healthy':
        return jsonify({
            'status': 'healthy',
            'timestamp': datetime.utcnow().isoformat(),
            'gpu_available': GPU_AVAILABLE,
            'database_type': DATABASE_TYPE
        }), 200
    elif app_health_state == 'initializing':
        return '', 204  # No content - still initializing
    else:
        return jsonify({
            'status': 'unhealthy',
            'timestamp': datetime.utcnow().isoformat()
        }), 503

@app.route('/ready', methods=['GET'])
def ready():
    """Readiness probe - checks if database is initialized"""
    if db_initialized:
        return jsonify({
            'status': 'ready',
            'database': 'initialized',
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
            'debug_scenario': int(request.form.get('debug_scenario')) if request.form.get('debug_scenario') else None
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
        
        # Validate RunPod configuration
        if not RUNPOD_ENDPOINT_ID or not RUNPOD_API_KEY:
            return jsonify({
                'error': 'RunPod worker not configured',
                'message': 'RUNPOD_ENDPOINT_ID and RUNPOD_API_KEY must be set in environment variables'
            }), 500
        
        # Create job in database
        create_job(job_id, parameters, uploaded_files)
        
        # Start RunPod worker job in background thread
        thread = threading.Thread(target=trigger_runpod_job, args=(job_id,))
        thread.daemon = True
        thread.start()
        job_threads[job_id] = thread
        
        return jsonify({
            'job_id': job_id,
            'status': 'pending',
            'message': 'Job created and sent to RunPod worker',
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
    Get job results from database
    Query params:
        type: 'summary' (default), 'detailed', or 'internal'
        format: 'json' (default) or 'csv'
        an_eval: Filter by year (for internal type)
        mois_eval: Filter by month (for internal type)
        id_compte: Filter by account ID (for detailed type)
    Returns data for the specified result type
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
        
        # Get result type and format from query params
        result_type = request.args.get('type', 'summary').lower()
        result_format = request.args.get('format', 'json').lower()
        
        # Get filter parameters
        an_eval = request.args.get('an_eval', type=int)
        mois_eval = request.args.get('mois_eval', type=int)
        id_compte = request.args.get('id_compte', type=int)
        
        # Retrieve data from database with filters
        filters_applied = []
        if result_type == 'summary':
            df = get_vp_flux_total(job_id)
        elif result_type == 'detailed':
            df = get_vp_flux_compte(job_id, id_compte=id_compte)
            if id_compte is not None:
                filters_applied.append(f'id_compte={id_compte}')
        elif result_type == 'internal':
            df = get_flux_projetes(job_id, an_eval=an_eval, mois_eval=mois_eval)
            if an_eval is not None:
                filters_applied.append(f'an_eval={an_eval}')
            if mois_eval is not None:
                filters_applied.append(f'mois_eval={mois_eval}')
        else:
            return jsonify({
                'error': 'Invalid result type',
                'message': 'Type must be one of: summary, detailed, internal'
            }), 400
        
        if df is None or len(df) == 0:
            return jsonify({
                'error': 'No results found',
                'message': 'Results not available in database for this job'
            }), 404
        
        # Return in requested format
        if result_format == 'csv':
            # Return as CSV download
            from io import StringIO
            output = StringIO()
            df.to_csv(output, sep=';', index=False)
            output.seek(0)
            return output.getvalue(), 200, {
                'Content-Type': 'text/csv',
                'Content-Disposition': f'attachment; filename={result_type}_{job_id}.csv'
            }
        else:
            # Return as JSON
            data = df.to_dict(orient='records')
            response = {
                'job_id': job_id,
                'result_type': result_type,
                'count': len(data),
                'data': data
            }
            if filters_applied:
                response['filters'] = filters_applied
            return jsonify(response)
        
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
        with get_db_cursor() as (cursor, conn):
            cursor.execute("DELETE FROM jobs")
            deleted_count = cursor.rowcount
        
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
    print("=" * 60)
    print("GPU Actuarial Projections API")
    print("=" * 60)
    print(f"Version: {APP_VERSION}")
    print(f"Environment: {ENVIRONMENT}")
    print(f"GPU Available: {GPU_AVAILABLE}")
    print(f"Database Type: {DATABASE_TYPE}")
    print(f"Database: {app.config['DATABASE']}")
    print(f"Upload Folder: {app.config['UPLOAD_FOLDER']}")
    print(f"Results Folder: {app.config['RESULTS_FOLDER']}")
    print(f"Admin Password: {'***' if ENVIRONMENT == 'production' else ADMIN_PASSWORD}")
    print(f"Port (Main): {PORT}")
    if PORT_HEALTH != PORT:
        print(f"Port (Health): {PORT_HEALTH}")
    print("=" * 60)
    if ENVIRONMENT != 'production':
        print("⚠️  WARNING: Using default admin password!")
        print("⚠️  Set ADMIN_PASSWORD environment variable for production")
        print("=" * 60)
    
    # RunPod load balancing compatibility note
    if PORT == 80:
        print("ℹ️  Running in RunPod-compatible mode (PORT=80)")
        print("ℹ️  Health check endpoint: /ping")
        print("=" * 60)
    
    # Run the app
    app.run(
        host='0.0.0.0',
        port=PORT,
        debug=(ENVIRONMENT == 'development')
    )
