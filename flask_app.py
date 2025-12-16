"""
Flask API for GPU-based Actuarial Projections
Provides RESTful endpoints for job management and result retrieval
"""

import os
import sqlite3
import json
import threading
import traceback
import io
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
    from calculations.gpu import run_projection_gpu_nested, ProjectionResult
    GPU_AVAILABLE = True
except Exception as e:
    print(f"Warning: GPU module not available: {e}")
    GPU_AVAILABLE = False

# =============================================================================
# KERNEL PATCHING SUPPORT
# =============================================================================

import importlib

# Store original kernel content for restoration
KERNELS_PATH = HERE / 'calculations' / 'kernels.py'
_original_kernel_content = None
_kernel_lock = threading.Lock()


def apply_custom_kernel(kernel_code: str) -> dict:
    """
    Apply custom kernel code by writing to kernels.py and reloading modules.
    Thread-safe with locking.
    
    Returns dict with 'success' or 'error' key.
    """
    global _original_kernel_content
    
    with _kernel_lock:
        try:
            # Backup original content (only once)
            if _original_kernel_content is None and KERNELS_PATH.exists():
                _original_kernel_content = KERNELS_PATH.read_text()
            
            # Validate syntax first
            try:
                compile(kernel_code, 'kernels.py', 'exec')
            except SyntaxError as e:
                return {
                    'error': f'Syntax error at line {e.lineno}: {e.msg}',
                    'line': e.lineno
                }
            
            # Write new kernel code
            KERNELS_PATH.write_text(kernel_code)
            print(f"[KERNEL] Custom kernel code written ({len(kernel_code)} bytes)")
            
            # Reload modules
            import calculations.kernels
            importlib.reload(calculations.kernels)
            print("[KERNEL] Reloaded calculations.kernels")
            
            import calculations.gpu
            importlib.reload(calculations.gpu)
            print("[KERNEL] Reloaded calculations.gpu")
            
            return {'success': True}
            
        except Exception as e:
            # Restore original on failure
            restore_original_kernel()
            return {
                'error': f'Failed to apply custom kernel: {str(e)}',
                'traceback': traceback.format_exc()
            }


def restore_original_kernel():
    """Restore the original kernel code. Thread-safe."""
    global _original_kernel_content
    
    with _kernel_lock:
        if _original_kernel_content is not None:
            try:
                KERNELS_PATH.write_text(_original_kernel_content)
                
                import calculations.kernels
                importlib.reload(calculations.kernels)
                
                import calculations.gpu
                importlib.reload(calculations.gpu)
                
                print("[KERNEL] Restored original kernel code")
            except Exception as e:
                print(f"[KERNEL] Warning: Failed to restore original kernel: {e}")

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
USE_NEONDB = os.getenv('USE_NEONDB', 'true').lower() == 'true'
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
            vp_reserve_be REAL,
            vp_capital_req REAL,
            vp_scr REAL,
            avg_reserve_be REAL,
            avg_capital_req REAL,
            avg_scr REAL,
            n_accounts INTEGER,
            FOREIGN KEY (job_id) REFERENCES jobs(job_id) ON DELETE CASCADE
        )
    """
    
    # New tables for nested stochastic results
    nested_results_table = f"""
        CREATE TABLE IF NOT EXISTS nested_results (
            id {id_column},
            job_id TEXT NOT NULL,
            id_compte INTEGER NOT NULL,
            reserve_be REAL,
            capital_req REAL,
            scr REAL,
            FOREIGN KEY (job_id) REFERENCES jobs(job_id) ON DELETE CASCADE
        )
    """
    
    nested_summary_table = f"""
        CREATE TABLE IF NOT EXISTS nested_summary (
            id {id_column},
            job_id TEXT NOT NULL,
            categorie TEXT NOT NULL,
            vp_reserve_be REAL,
            vp_capital_req REAL,
            vp_scr REAL,
            avg_reserve_be REAL,
            avg_capital_req REAL,
            avg_scr REAL,
            n_accounts INTEGER,
            FOREIGN KEY (job_id) REFERENCES jobs(job_id) ON DELETE CASCADE
        )
    """
    
    five_chocs_results_table = f"""
        CREATE TABLE IF NOT EXISTS five_chocs_results (
            id {id_column},
            job_id TEXT NOT NULL,
            id_compte INTEGER NOT NULL,
            choc_type TEXT NOT NULL,
            reserve_be REAL,
            capital_req REAL,
            scr REAL,
            FOREIGN KEY (job_id) REFERENCES jobs(job_id) ON DELETE CASCADE
        )
    """
    
    sensitivities_table = f"""
        CREATE TABLE IF NOT EXISTS sensitivities (
            id {id_column},
            job_id TEXT NOT NULL,
            id_compte INTEGER NOT NULL,
            delta_sp500_reserve REAL,
            delta_tsx_reserve REAL,
            delta_eafe_reserve REAL,
            delta_dex_reserve REAL,
            delta_sp500_capital REAL,
            delta_tsx_capital REAL,
            delta_eafe_capital REAL,
            delta_dex_capital REAL,
            FOREIGN KEY (job_id) REFERENCES jobs(job_id) ON DELETE CASCADE
        )
    """
    
    chocs_summary_table = f"""
        CREATE TABLE IF NOT EXISTS chocs_summary (
            id {id_column},
            job_id TEXT NOT NULL,
            choc_type TEXT NOT NULL,
            reserve_be_sum REAL,
            reserve_be_mean REAL,
            capital_req_sum REAL,
            capital_req_mean REAL,
            scr_sum REAL,
            scr_mean REAL,
            FOREIGN KEY (job_id) REFERENCES jobs(job_id) ON DELETE CASCADE
        )
    """
    
    ext_debug_table = f"""
        CREATE TABLE IF NOT EXISTS ext_debug (
            id {id_column},
            job_id TEXT NOT NULL,
            debug_account INTEGER,
            debug_scenario INTEGER,
            debug_year INTEGER,
            debug_month INTEGER,
            vm REAL,
            age REAL,
            qx REAL,
            lapse_tot REAL,
            lapse_part REAL,
            tx_survie REAL,
            forward_rate REAL,
            rend_sp500 REAL,
            rend_tsx REAL,
            rend_eafe REAL,
            rend_dex REAL,
            retrait REAL,
            prest_deces REAL,
            primes_garanties REAL,
            vm_vg_ratio REAL,
            FOREIGN KEY (job_id) REFERENCES jobs(job_id) ON DELETE CASCADE
        )
    """
    
    int_debug_table = f"""
        CREATE TABLE IF NOT EXISTS int_debug (
            id {id_column},
            job_id TEXT NOT NULL,
            choc_idx INTEGER,
            choc_name TEXT,
            debug_int_scenario INTEGER,
            debug_int_year INTEGER,
            start_vm REAL,
            vm_choc REAL,
            avg_pv_flux REAL,
            reserve REAL,
            capital REAL,
            start_tx_survie REAL,
            start_age REAL,
            int_curr_vm REAL,
            int_fees REAL,
            int_pv_path REAL,
            int_r_portfolio REAL,
            int_fwd_rate REAL,
            FOREIGN KEY (job_id) REFERENCES jobs(job_id) ON DELETE CASCADE
        )
    """
    
    # Kernel versions table for storing kernel.py history
    kernel_versions_table = f"""
        CREATE TABLE IF NOT EXISTS kernel_versions (
            id {id_column},
            version_name TEXT NOT NULL,
            created_at TEXT NOT NULL,
            description TEXT,
            content TEXT NOT NULL,
            is_active INTEGER DEFAULT 0
        )
    """
    
    with get_db_cursor() as (cursor, conn):
        # Create tables
        cursor.execute(jobs_table)
        cursor.execute(flux_projetes_table)
        cursor.execute(vp_flux_compte_table)
        cursor.execute(vp_flux_total_table)
        cursor.execute(nested_results_table)
        cursor.execute(nested_summary_table)
        cursor.execute(five_chocs_results_table)
        cursor.execute(sensitivities_table)
        cursor.execute(chocs_summary_table)
        cursor.execute(ext_debug_table)
        cursor.execute(int_debug_table)
        cursor.execute(kernel_versions_table)
        
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
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_nested_results_job_id 
            ON nested_results(job_id)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_nested_summary_job_id 
            ON nested_summary(job_id)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_five_chocs_results_job_id 
            ON five_chocs_results(job_id)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_sensitivities_job_id 
            ON sensitivities(job_id)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_chocs_summary_job_id 
            ON chocs_summary(job_id)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_ext_debug_job_id 
            ON ext_debug(job_id)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_int_debug_job_id 
            ON int_debug(job_id)
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

def update_job_progress(job_id: str, current_batch: int, total_batches: int, progress_percent: int = None) -> None:
    """
    Update job progress
    
    Args:
        job_id: Job identifier
        current_batch: Current batch number (1-indexed)
        total_batches: Total number of batches
        progress_percent: Optional override for progress percentage
    """
    if progress_percent is None:
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

def save_nested_results(job_id: str, df: pd.DataFrame) -> None:
    """Save nested stochastic results (per-account reserves/capital) to database table"""
    engine = get_sqlalchemy_engine()
    
    # Prepare data for bulk insert
    df_copy = df.copy()
    df_copy.insert(0, 'job_id', job_id)
    
    # Rename columns to match database schema (lowercase)
    df_copy.columns = df_copy.columns.str.lower()
    
    # Insert into database
    df_copy.to_sql('nested_results', engine, if_exists='append', index=False)
    engine.dispose()
    print(f"  Saved {len(df)} nested_results records to database")

def save_nested_summary(job_id: str, df: pd.DataFrame) -> None:
    """Save nested stochastic summary (portfolio totals) to database table"""
    engine = get_sqlalchemy_engine()
    
    # Prepare data for bulk insert
    df_copy = df.copy()
    df_copy.insert(0, 'job_id', job_id)
    
    # Rename columns to match database schema (lowercase)
    df_copy.columns = df_copy.columns.str.lower()
    
    # Insert into database
    df_copy.to_sql('nested_summary', engine, if_exists='append', index=False)
    engine.dispose()
    print(f"  Saved {len(df)} nested_summary records to database")

def save_five_chocs_results(job_id: str, df: pd.DataFrame) -> None:
    """Save five chocs results (per-account per-choc) to database table"""
    engine = get_sqlalchemy_engine()
    df_copy = df.copy()
    df_copy.insert(0, 'job_id', job_id)
    df_copy.columns = df_copy.columns.str.lower()
    df_copy.to_sql('five_chocs_results', engine, if_exists='append', index=False)
    engine.dispose()
    print(f"  Saved {len(df)} five_chocs_results records to database")

def save_sensitivities(job_id: str, df: pd.DataFrame) -> None:
    """Save sensitivities/Greeks (per-account deltas) to database table"""
    engine = get_sqlalchemy_engine()
    df_copy = df.copy()
    df_copy.insert(0, 'job_id', job_id)
    df_copy.columns = df_copy.columns.str.lower()
    df_copy.to_sql('sensitivities', engine, if_exists='append', index=False)
    engine.dispose()
    print(f"  Saved {len(df)} sensitivities records to database")

def save_chocs_summary(job_id: str, df: pd.DataFrame) -> None:
    """Save chocs summary (aggregated by choc type) to database table"""
    engine = get_sqlalchemy_engine()
    df_copy = df.copy()
    df_copy.insert(0, 'job_id', job_id)
    df_copy.columns = df_copy.columns.str.lower()
    df_copy.to_sql('chocs_summary', engine, if_exists='append', index=False)
    engine.dispose()
    print(f"  Saved {len(df)} chocs_summary records to database")

def save_ext_debug(job_id: str, df: pd.DataFrame) -> None:
    """Save external kernel debug output to database table"""
    engine = get_sqlalchemy_engine()
    df_copy = df.copy()
    df_copy.insert(0, 'job_id', job_id)
    df_copy.columns = df_copy.columns.str.lower()
    df_copy.to_sql('ext_debug', engine, if_exists='append', index=False)
    engine.dispose()
    print(f"  Saved {len(df)} ext_debug records to database")

def save_int_debug(job_id: str, df: pd.DataFrame) -> None:
    """Save internal kernel debug output to database table"""
    engine = get_sqlalchemy_engine()
    df_copy = df.copy()
    df_copy.insert(0, 'job_id', job_id)
    df_copy.columns = df_copy.columns.str.lower()
    df_copy.to_sql('int_debug', engine, if_exists='append', index=False)
    engine.dispose()
    print(f"  Saved {len(df)} int_debug records to database")

def get_nested_results(job_id: str, id_compte: int = None) -> Optional[pd.DataFrame]:
    """
    Retrieve nested stochastic results for a job with optional filter
    
    Args:
        job_id: Job identifier
        id_compte: Filter by account ID (optional)
    """
    conn = get_db_connection()
    try:
        ph = get_placeholder()
        query = f"SELECT * FROM nested_results WHERE job_id = {ph}"
        params = [job_id]
        
        if id_compte is not None:
            query += f" AND id_compte = {ph}"
            params.append(id_compte)
        
        query += " ORDER BY id_compte"
        
        df = pd.read_sql_query(query, conn, params=params)
        conn.close()
        if len(df) > 0:
            df = df.drop(columns=['id', 'job_id'])
            df.columns = df.columns.str.upper()
            return df
        return None
    except Exception as e:
        conn.close()
        print(f"Error retrieving nested_results: {e}")
        return None

def get_nested_summary(job_id: str) -> Optional[pd.DataFrame]:
    """Retrieve nested stochastic summary for a job"""
    conn = get_db_connection()
    try:
        ph = get_placeholder()
        df = pd.read_sql_query(
            f"SELECT * FROM nested_summary WHERE job_id = {ph}",
            conn,
            params=(job_id,)
        )
        conn.close()
        if len(df) > 0:
            df = df.drop(columns=['id', 'job_id'])
            df.columns = df.columns.str.upper()
            return df
        return None
    except Exception as e:
        conn.close()
        print(f"Error retrieving nested_summary: {e}")
        return None

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
                    # Show generic progress message with elapsed time
                    # Note: Do NOT call run_request.output() here as it blocks until completion
                    minutes = int(elapsed_time / 60)
                    seconds = int(elapsed_time % 60)
                    msg = f"🚀 GPU projection in progress... ({minutes}m {seconds}s elapsed)"
                    update_job_status(job_id, 'running', progress_message=msg)
                
                if status == "COMPLETED":
                    # Get the output
                    try:
                        output = run_request.output()
                        print(f"  RunPod job completed! Raw output type: {type(output)}")
                        print(f"  Output keys: {output.keys() if isinstance(output, dict) else 'N/A'}")
                    except Exception as e:
                        error_msg = f"Failed to retrieve RunPod output: {str(e)}"
                        print(f"✗ {error_msg}")
                        update_job_status(job_id, 'failed', error_message=error_msg)
                        return
                    
                    # Check if output contains results
                    if output and isinstance(output, dict):
                        if 'results' in output:
                            # Convert results to DataFrame format and save to database
                            results_data = output['results']
                            print(f"  Processing results from RunPod worker...")
                            print(f"  Results data keys: {results_data.keys() if isinstance(results_data, dict) else 'N/A'}")
                            
                            # Save legacy JSON format
                            update_job_results_data(job_id, output)
                            
                            # Convert JSON results back to DataFrames and save to proper tables
                            saved_any = False
                            if isinstance(results_data, dict):
                                # Save per-account results (reserves/capital/SCR)
                                if results_data.get('results'):
                                    try:
                                        df = pd.DataFrame(results_data['results'])
                                        save_nested_results(job_id, df)
                                        saved_any = True
                                        print(f"  ✓ Saved nested_results: {len(df)} rows")
                                    except Exception as e:
                                        print(f"  ✗ Failed to save nested_results: {e}")
                                
                                # Save portfolio summary (vp_flux_total)
                                if results_data.get('vp_flux_total'):
                                    try:
                                        df = pd.DataFrame(results_data['vp_flux_total'])
                                        save_nested_summary(job_id, df)
                                        saved_any = True
                                        total_pv = df['VP_RESERVE_BE'].iloc[0] if 'VP_RESERVE_BE' in df.columns else 0.0
                                        print(f"  ✓ Saved nested_summary (vp_flux_total): Total Reserve BE = ${total_pv:,.2f}")
                                    except Exception as e:
                                        print(f"  ✗ Failed to save nested_summary: {e}")
                                
                                # Save five chocs results
                                if results_data.get('results_5chocs'):
                                    try:
                                        df = pd.DataFrame(results_data['results_5chocs'])
                                        save_five_chocs_results(job_id, df)
                                        saved_any = True
                                        print(f"  ✓ Saved five_chocs_results: {len(df)} rows")
                                    except Exception as e:
                                        print(f"  ✗ Failed to save five_chocs_results: {e}")
                                
                                # Save sensitivities (Greeks/deltas)
                                if results_data.get('sensitivities'):
                                    try:
                                        df = pd.DataFrame(results_data['sensitivities'])
                                        save_sensitivities(job_id, df)
                                        saved_any = True
                                        print(f"  ✓ Saved sensitivities: {len(df)} rows")
                                    except Exception as e:
                                        print(f"  ✗ Failed to save sensitivities: {e}")
                                
                                # Save chocs summary
                                if results_data.get('chocs_summary'):
                                    try:
                                        df = pd.DataFrame(results_data['chocs_summary'])
                                        save_chocs_summary(job_id, df)
                                        saved_any = True
                                        print(f"  ✓ Saved chocs_summary: {len(df)} rows")
                                    except Exception as e:
                                        print(f"  ✗ Failed to save chocs_summary: {e}")
                                
                                # Save external debug output
                                if results_data.get('ext_debug'):
                                    try:
                                        df = pd.DataFrame(results_data['ext_debug'])
                                        save_ext_debug(job_id, df)
                                        saved_any = True
                                        print(f"  ✓ Saved ext_debug: {len(df)} rows")
                                    except Exception as e:
                                        print(f"  ✗ Failed to save ext_debug: {e}")
                                
                                # Save internal debug output
                                if results_data.get('int_debug'):
                                    try:
                                        df = pd.DataFrame(results_data['int_debug'])
                                        save_int_debug(job_id, df)
                                        saved_any = True
                                        print(f"  ✓ Saved int_debug: {len(df)} rows")
                                    except Exception as e:
                                        print(f"  ✗ Failed to save int_debug: {e}")
                                
                                # Save flux_projetes (FLUX_PROJETES_GPU.csv data)
                                if results_data.get('flux_projetes'):
                                    try:
                                        df = pd.DataFrame(results_data['flux_projetes'])
                                        save_flux_projetes(job_id, df)
                                        saved_any = True
                                        print(f"  ✓ Saved flux_projetes: {len(df)} rows")
                                    except Exception as e:
                                        print(f"  ✗ Failed to save flux_projetes: {e}")
                            
                            if saved_any:
                                print(f"✓ Job {job_id} completed and results saved to database!")
                            else:
                                print(f"⚠ Job {job_id} completed but no DataFrame results were saved")
                                print(f"  Results data structure: {type(results_data)}")
                            
                            # Clear progress message and mark as completed
                            ph = get_placeholder()
                            sql = f"UPDATE jobs SET status = {ph}, completed_at = {ph}, error_message = NULL WHERE job_id = {ph}"
                            execute_sql(sql, ('completed', datetime.utcnow().isoformat(), job_id))
                            print(f"  ✓ Job status updated to completed")
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
        if params.get('nb_int_scenarios'):
            runpod_input['nb_int_scenarios'] = params.get('nb_int_scenarios')
        if params.get('shock_capital_pct') is not None:
            runpod_input['shock_capital_pct'] = params.get('shock_capital_pct')
        if params.get('max_accounts'):
            runpod_input['max_accounts'] = params.get('max_accounts')
        if params.get('debug_account'):
            runpod_input['debug_account'] = params.get('debug_account')
        if params.get('debug_scenario'):
            runpod_input['debug_scenario'] = params.get('debug_scenario')
        if params.get('debug_year'):
            runpod_input['debug_year'] = params.get('debug_year')
        if params.get('debug_month'):
            runpod_input['debug_month'] = params.get('debug_month')
        if params.get('debug_int_scenario'):
            runpod_input['debug_int_scenario'] = params.get('debug_int_scenario')
        if params.get('debug_int_year'):
            runpod_input['debug_int_year'] = params.get('debug_int_year')
        if params.get('debug_only'):
            runpod_input['debug_only'] = params.get('debug_only')
        
        # Add custom kernel code if provided
        if params.get('kernel_code'):
            runpod_input['kernel_code'] = params.get('kernel_code')
            print(f"  Including custom kernel code ({len(params['kernel_code'])} bytes)")

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
    custom_kernel_applied = False
    
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
        nb_ext_scenarios = params.get('nb_scenarios', 100)  # External scenarios (real-world)
        nb_int_scenarios = params.get('nb_int_scenarios', 100)  # Internal scenarios (risk-neutral)
        shock_capital_pct = params.get('shock_capital_pct', 0.35)  # Capital shock percentage
        max_accounts = params.get('max_accounts', None)
        debug_account = params.get('debug_account', -1) if params.get('debug_account') is not None else -1
        debug_scenario = params.get('debug_scenario', -1) if params.get('debug_scenario') is not None else -1
        debug_year = params.get('debug_year', -1) if params.get('debug_year') is not None else -1
        debug_month = params.get('debug_month', -1) if params.get('debug_month') is not None else -1
        debug_int_scenario = params.get('debug_int_scenario', -1) if params.get('debug_int_scenario') is not None else -1
        debug_int_year = params.get('debug_int_year', -1) if params.get('debug_int_year') is not None else -1
        debug_only = params.get('debug_only', False)
        
        # Check for custom kernel code
        kernel_code = params.get('kernel_code')
        if kernel_code:
            print(f"[KERNEL] Custom kernel code provided for job {job_id} ({len(kernel_code)} bytes)")
            result = apply_custom_kernel(kernel_code)
            if 'error' in result:
                raise Exception(f"Failed to apply custom kernel: {result['error']}")
            custom_kernel_applied = True
            print(f"[KERNEL] Custom kernel applied successfully for job {job_id}")
        
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
        if custom_kernel_applied:
            print(f"  Using custom kernel code")
        
        # Import from (possibly reloaded) module
        from calculations.gpu import run_projection_gpu_nested as run_projection
        
        results = run_projection(
            data_path=data_path,
            output_path=output_path,
            nb_an_projection=nb_years,
            nb_ext_scenarios=nb_ext_scenarios,
            nb_int_scenarios=nb_int_scenarios,
            shock_capital_pct=shock_capital_pct,
            max_accounts=max_accounts,
            debug_account=debug_account,
            debug_scenario=debug_scenario,
            debug_year=debug_year,
            debug_month=debug_month,
            debug_int_scenario=debug_int_scenario,
            debug_int_year=debug_int_year,
            debug_only=debug_only,
            progress_callback=progress_callback,
            **custom_paths
        )
        
        # Save results data to separate database tables
        # Note: run_projection_gpu_nested returns a ProjectionResult dataclass
        if results:
            print(f"\nSaving results to database...")
            # Save per-account results (reserves/capital/SCR)
            if results.results is not None:
                save_nested_results(job_id, results.results)
            
            # Save portfolio summary
            if results.vp_flux_total is not None:
                save_nested_summary(job_id, results.vp_flux_total)
            
            # Also save summary to JSON for backward compatibility
            results_data = {
                'vp_flux_total_summary': {
                    'vp_reserve_be': float(results.vp_flux_total['VP_RESERVE_BE'].iloc[0]) if results.vp_flux_total is not None and len(results.vp_flux_total) > 0 else 0.0,
                    'vp_capital_req': float(results.vp_flux_total['VP_CAPITAL_REQ'].iloc[0]) if results.vp_flux_total is not None and len(results.vp_flux_total) > 0 else 0.0,
                    'vp_scr': float(results.vp_flux_total['VP_SCR'].iloc[0]) if results.vp_flux_total is not None and len(results.vp_flux_total) > 0 else 0.0,
                },
                'nested_results_summary': {
                    'total_accounts': len(results.results) if results.results is not None else 0
                },
                'total_duration': results.total_duration,
                'saved_files': results.saved_files
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
        
        # Restore original kernel after successful completion
        if custom_kernel_applied:
            restore_original_kernel()
        
    except Exception as e:
        # Restore original kernel on error
        if custom_kernel_applied:
            restore_original_kernel()
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
        
        # Get parameters from form or JSON
        if request.is_json:
            form_data = request.json
        else:
            form_data = request.form
        
        parameters = {
            'nb_an_projection': int(form_data.get('nb_an_projection', 100)),
            'nb_scenarios': int(form_data.get('nb_scenarios', 100)),
            'nb_int_scenarios': int(form_data.get('nb_int_scenarios', 100)),
            'shock_capital_pct': float(form_data.get('shock_capital_pct', 35)) / 100.0,  # Convert % to decimal
            'max_accounts': int(form_data.get('max_accounts')) if form_data.get('max_accounts') else None,
            'debug_account': int(form_data.get('debug_account')) if form_data.get('debug_account') else None,
            'debug_scenario': int(form_data.get('debug_scenario')) if form_data.get('debug_scenario') else None,
            'debug_year': int(form_data.get('debug_year')) if form_data.get('debug_year') else None,
            'debug_month': int(form_data.get('debug_month')) if form_data.get('debug_month') else None,
            'debug_int_scenario': int(form_data.get('debug_int_scenario')) if form_data.get('debug_int_scenario') else None,
            'debug_int_year': int(form_data.get('debug_int_year')) if form_data.get('debug_int_year') else None,
            'debug_only': form_data.get('debug_only', '').lower() in ('true', '1', 'yes', 'on'),
        }
        
        # Add custom kernel code if provided
        kernel_code = form_data.get('kernel_code')
        if kernel_code:
            parameters['kernel_code'] = kernel_code
            print(f"Job {job_id}: Custom kernel code provided ({len(kernel_code)} bytes)")
        
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
        
        # Get result DataFrames from database tables
        result_files = []
        ph = get_placeholder()
        
        # Check for nested_summary (portfolio totals - vp_flux_total)
        try:
            sql = f"SELECT COUNT(*) as count FROM nested_summary WHERE job_id = {ph}"
            total_count = fetch_all(sql, (job_id,))
            if total_count and total_count[0]['count'] > 0:
                # Get the actual total value
                sql = f"SELECT vp_reserve_be, vp_capital_req, vp_scr FROM nested_summary WHERE job_id = {ph} LIMIT 1"
                total_val = fetch_all(sql, (job_id,))
                pv_value = total_val[0]['vp_reserve_be'] if total_val else 0
                result_files.append({
                    'name': 'VP_FLUX_TOTAL',
                    'type': 'summary',
                    'description': f'Total Reserve BE: ${pv_value:,.2f}',
                    'row_count': total_count[0]['count'],
                    'table': 'nested_summary',
                    'pv_total': pv_value
                })
        except Exception as e:
            print(f"Error checking nested_summary: {e}")
        
        # Check for nested_results (per-account reserves/capital)
        try:
            sql = f"SELECT COUNT(*) as count FROM nested_results WHERE job_id = {ph}"
            results_count = fetch_all(sql, (job_id,))
            if results_count and results_count[0]['count'] > 0:
                result_files.append({
                    'name': 'NESTED_RESULTS',
                    'type': 'detailed',
                    'description': f'Per-account reserves/capital ({results_count[0]["count"]} accounts)',
                    'row_count': results_count[0]['count'],
                    'table': 'nested_results'
                })
        except Exception as e:
            print(f"Error checking nested_results: {e}")
        
        # Check for five_chocs_results
        try:
            sql = f"SELECT COUNT(*) as count FROM five_chocs_results WHERE job_id = {ph}"
            chocs_count = fetch_all(sql, (job_id,))
            if chocs_count and chocs_count[0]['count'] > 0:
                result_files.append({
                    'name': 'FIVE_CHOCS_RESULTS',
                    'type': 'detailed',
                    'description': f'Five chocs per account ({chocs_count[0]["count"]} rows)',
                    'row_count': chocs_count[0]['count'],
                    'table': 'five_chocs_results'
                })
        except Exception as e:
            print(f"Error checking five_chocs_results: {e}")
        
        # Check for sensitivities (Greeks/deltas)
        try:
            sql = f"SELECT COUNT(*) as count FROM sensitivities WHERE job_id = {ph}"
            sens_count = fetch_all(sql, (job_id,))
            if sens_count and sens_count[0]['count'] > 0:
                result_files.append({
                    'name': 'SENSITIVITIES',
                    'type': 'detailed',
                    'description': f'Greeks/Deltas per account ({sens_count[0]["count"]} rows)',
                    'row_count': sens_count[0]['count'],
                    'table': 'sensitivities'
                })
        except Exception as e:
            print(f"Error checking sensitivities: {e}")
        
        # Check for chocs_summary
        try:
            sql = f"SELECT COUNT(*) as count FROM chocs_summary WHERE job_id = {ph}"
            chocs_sum_count = fetch_all(sql, (job_id,))
            if chocs_sum_count and chocs_sum_count[0]['count'] > 0:
                result_files.append({
                    'name': 'CHOCS_SUMMARY',
                    'type': 'summary',
                    'description': f'Chocs summary by type ({chocs_sum_count[0]["count"]} rows)',
                    'row_count': chocs_sum_count[0]['count'],
                    'table': 'chocs_summary'
                })
        except Exception as e:
            print(f"Error checking chocs_summary: {e}")
        
        # Check for ext_debug
        try:
            sql = f"SELECT COUNT(*) as count FROM ext_debug WHERE job_id = {ph}"
            ext_debug_count = fetch_all(sql, (job_id,))
            if ext_debug_count and ext_debug_count[0]['count'] > 0:
                result_files.append({
                    'name': 'EXT_DEBUG',
                    'type': 'debug',
                    'description': f'External kernel debug output ({ext_debug_count[0]["count"]} rows)',
                    'row_count': ext_debug_count[0]['count'],
                    'table': 'ext_debug'
                })
        except Exception as e:
            print(f"Error checking ext_debug: {e}")
        
        # Check for int_debug
        try:
            sql = f"SELECT COUNT(*) as count FROM int_debug WHERE job_id = {ph}"
            int_debug_count = fetch_all(sql, (job_id,))
            if int_debug_count and int_debug_count[0]['count'] > 0:
                result_files.append({
                    'name': 'INT_DEBUG',
                    'type': 'debug',
                    'description': f'Internal kernel debug output ({int_debug_count[0]["count"]} rows)',
                    'row_count': int_debug_count[0]['count'],
                    'table': 'int_debug'
                })
        except Exception as e:
            print(f"Error checking int_debug: {e}")

        results_folder = get_job_results_folder(job_id)
        if results_folder.exists():
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
                },
                'VP_FLUX_5CHOCS_DETAILED_GPU.csv': {
                    'type': 'detailed',
                    'description': 'Five chocs detailed results'
                },
                'VP_FLUX_5CHOCS_SUMMARY_GPU.csv': {
                    'type': 'summary',
                    'description': 'Five chocs summary'
                },
                'VP_FLUX_SENSITIVITIES_GPU.csv': {
                    'type': 'detailed',
                    'description': 'Sensitivities/Greeks by account'
                },
                'DEBUG_EXTERNAL_KERNEL.csv': {
                    'type': 'debug',
                    'description': 'External kernel debug output'
                },
                'DEBUG_INTERNAL_KERNEL.csv': {
                    'type': 'debug',
                    'description': 'Internal kernel debug output'
                },
            }

            for file in results_folder.glob('*.csv'):
                file_info = {
                    'name': file.name,
                    'size': file.stat().st_size,
                    'type': 'other',
                    'description': 'Output CSV'
                }
                if file.name in output_file_metadata:
                    file_info.update(output_file_metadata[file.name])
                result_files.append(file_info)
        
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
        
        # Map display names to database tables
        table_map = {
            'FLUX_PROJETES': 'flux_projetes',
            'VP_FLUX_COMPTE': 'vp_flux_compte',
            'VP_FLUX_TOTAL': 'nested_summary',
            'NESTED_RESULTS': 'nested_results',
            'FIVE_CHOCS_RESULTS': 'five_chocs_results',
            'SENSITIVITIES': 'sensitivities',
            'CHOCS_SUMMARY': 'chocs_summary',
            'EXT_DEBUG': 'ext_debug',
            'INT_DEBUG': 'int_debug'
        }
        
        table_name = table_map.get(file_name.upper())
        
        if table_name:
            # Fetch from database table
            ph = get_placeholder()
            sql = f"SELECT * FROM {table_name} WHERE job_id = {ph} LIMIT {limit}"
            rows = fetch_all(sql, (job_id,))
            
            if not rows:
                return jsonify({
                    'error': 'No data found',
                    'file_name': file_name
                }), 404
            
            # Get total count
            sql = f"SELECT COUNT(*) as count FROM {table_name} WHERE job_id = {ph}"
            count_result = fetch_all(sql, (job_id,))
            total_rows = count_result[0]['count'] if count_result else 0
            
            # Convert to DataFrame for consistent handling
            df = pd.DataFrame(rows)
            # Remove job_id column for display
            if 'job_id' in df.columns:
                df = df.drop(columns=['job_id'])
            
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
        else:
            # Fall back to file-based results (for uploaded input files and job output files)
            file_name = secure_filename(file_name)
            upload_folder = get_job_upload_folder(job_id)
            file_path = upload_folder / file_name

            if not file_path.exists():
                results_folder = get_job_results_folder(job_id)
                file_path = results_folder / file_name
            
            if not file_path.exists():
                return jsonify({
                    'error': 'File not found',
                    'file_name': file_name
                }), 404
            
            # Read CSV with limit
            df = pd.read_csv(file_path, sep=';', nrows=limit)
            total_rows = sum(1 for _ in open(file_path)) - 1
            
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
        
        # Map display names to database tables
        table_map = {
            'FLUX_PROJETES': 'flux_projetes',
            'VP_FLUX_COMPTE': 'vp_flux_compte',
            'VP_FLUX_TOTAL': 'nested_summary',
            'NESTED_RESULTS': 'nested_results',
            'FIVE_CHOCS_RESULTS': 'five_chocs_results',
            'SENSITIVITIES': 'sensitivities',
            'CHOCS_SUMMARY': 'chocs_summary',
            'EXT_DEBUG': 'ext_debug',
            'INT_DEBUG': 'int_debug'
        }
        
        table_name = table_map.get(file_name.upper())
        
        if table_name:
            # Generate CSV from database table
            ph = get_placeholder()
            sql = f"SELECT * FROM {table_name} WHERE job_id = {ph}"
            rows = fetch_all(sql, (job_id,))
            
            if not rows:
                return jsonify({
                    'error': 'No data found',
                    'file_name': file_name
                }), 404
            
            # Convert to DataFrame and remove job_id column
            df = pd.DataFrame(rows)
            if 'job_id' in df.columns:
                df = df.drop(columns=['job_id'])
            
            # Create CSV in memory
            from io import StringIO
            output = StringIO()
            df.to_csv(output, index=False, sep=';')
            output.seek(0)
            
            # Return as downloadable file
            return send_file(
                io.BytesIO(output.getvalue().encode('utf-8')),
                mimetype='text/csv',
                as_attachment=True,
                download_name=f'{file_name}.csv'
            )
        else:
            # Check for uploaded input files
            file_name = secure_filename(file_name)
            upload_file = get_job_upload_folder(job_id) / file_name
            if upload_file.exists():
                return send_file(upload_file, mimetype='text/plain', as_attachment=True)

            results_file = get_job_results_folder(job_id) / file_name
            if results_file.exists():
                return send_file(results_file, mimetype='text/csv', as_attachment=True)
            
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
# API ROUTES - KERNEL MANAGEMENT
# =============================================================================

@app.route('/admin/kernels/file', methods=['GET'])
def get_kernel_file():
    """
    Get the current kernels.py file content.
    """
    kernels_path = HERE / 'calculations' / 'kernels.py'
    if not kernels_path.exists():
        return jsonify({'error': 'kernels.py not found'}), 404
    
    return jsonify({
        'path': str(kernels_path),
        'content': kernels_path.read_text()
    })

@app.route('/admin/kernels/validate', methods=['POST'])
def validate_kernel_file():
    """
    Validate kernel code by running a test projection on RunPod.
    
    JSON body:
    {
        "password": "admin123",
        "content": "... new kernels.py content ..."
    }
    
    Returns validation result. This is a synchronous call that waits for RunPod to complete.
    """
    if not request.is_json:
        return jsonify({'error': 'JSON body required'}), 400
    
    data = request.json
    password = data.get('password')
    if password != ADMIN_PASSWORD:
        return jsonify({'error': 'Invalid password'}), 403
    
    content = data.get('content')
    if not content:
        return jsonify({'error': 'content is required'}), 400
    
    # Step 1: Check Python syntax locally (fast check)
    try:
        compile(content, 'kernels.py', 'exec')
    except SyntaxError as e:
        return jsonify({
            'valid': False,
            'stage': 'syntax',
            'error': f'Syntax error at line {e.lineno}: {e.msg}',
            'line': e.lineno,
            'offset': e.offset
        }), 400
    
    # Step 2: Run validation on RunPod with the custom kernel
    if not RUNPOD_API_KEY or not RUNPOD_ENDPOINT_ID:
        return jsonify({
            'valid': False,
            'stage': 'configuration',
            'error': 'RunPod API key or endpoint ID not configured. Cannot validate kernel remotely.'
        }), 500
    
    try:
        print(f"[KERNEL VALIDATION] Triggering RunPod validation job...")
        
        # Create a minimal validation job payload
        runpod_input = {
            'nb_an_projection': 10,      # Short projection
            'nb_scenarios': 2,           # Minimal external scenarios
            'nb_int_scenarios': 2,       # Minimal internal scenarios
            'max_accounts': 1,           # Just 1 account
            'kernel_code': content,      # The custom kernel to validate
            'data_file_urls': {}         # Use default data files on worker
        }
        
        endpoint = runpod.Endpoint(RUNPOD_ENDPOINT_ID)
        
        # Check endpoint health first
        try:
            health = endpoint.health()
            print(f"  Endpoint health: {health}")
            if health.get('workers', {}).get('idle', 0) == 0 and health.get('workers', {}).get('running', 0) == 0:
                return jsonify({
                    'valid': False,
                    'stage': 'runpod',
                    'error': 'No RunPod workers available. Please try again later or start a worker.'
                }), 503
        except Exception as health_err:
            print(f"  Warning: Could not check endpoint health: {health_err}")
        
        # Run synchronously using run_sync which blocks until completion
        print(f"  Submitting validation job to RunPod (synchronous with 300s timeout)...")
        
        try:
            # run_sync blocks until the job completes or times out
            # timeout is in seconds (5 minutes should be plenty for a minimal validation)
            output = endpoint.run_sync(runpod_input, timeout=300)
            print(f"  RunPod job completed. Output type: {type(output)}")
            
            # Check for errors in output
            if isinstance(output, dict) and 'error' in output:
                error_msg = output['error']
                tb = output.get('traceback', '')
                
                # Check if it's a kernel-specific error
                if 'kernel_error' in output:
                    kernel_err = output['kernel_error']
                    return jsonify({
                        'valid': False,
                        'stage': 'kernel_load',
                        'error': kernel_err.get('error', error_msg),
                        'line': kernel_err.get('line'),
                        'traceback': kernel_err.get('traceback', tb)
                    }), 400
                
                return jsonify({
                    'valid': False,
                    'stage': 'runtime',
                    'error': error_msg,
                    'traceback': tb
                }), 400
            
            # Success!
            print(f"  ✓ Validation completed successfully on RunPod")
            return jsonify({
                'valid': True,
                'message': 'Kernel code validated successfully on RunPod GPU! Test projection completed with 1 account.'
            })
            
        except TimeoutError:
            return jsonify({
                'valid': False,
                'stage': 'timeout',
                'error': 'Validation timed out after 300 seconds. The RunPod worker may be busy or unavailable.'
            }), 504
        
    except Exception as e:
        return jsonify({
            'valid': False,
            'stage': 'validation',
            'error': f'Validation failed: {str(e)}',
            'traceback': traceback.format_exc()
        }), 500

@app.route('/admin/kernels/file', methods=['POST'])
def update_kernel_file():
    """
    Replace kernels.py content, save to database, and restart the Flask server.
    
    JSON body:
    {
        "password": "admin123",
        "content": "... new kernels.py content ...",
        "version_name": "v1.0",  // optional, auto-generated if not provided
        "description": "Fixed lapse calculation",  // optional
        "restart": true,  // optional, default true
        "validate": true  // optional, default true - run test before saving
    }
    
    WARNING: This overwrites the entire kernels.py file!
    """
    if not request.is_json:
        return jsonify({'error': 'JSON body required'}), 400
    
    data = request.json
    password = data.get('password')
    if password != ADMIN_PASSWORD:
        return jsonify({'error': 'Invalid password'}), 403
    
    content = data.get('content')
    if not content:
        return jsonify({'error': 'content is required'}), 400
    
    # Check if validation is requested (default: True)
    should_validate = data.get('validate', True)
    
    if should_validate:
        # Step 1: Check Python syntax locally (fast check)
        try:
            compile(content, 'kernels.py', 'exec')
        except SyntaxError as e:
            return jsonify({
                'error': f'Syntax error at line {e.lineno}: {e.msg}',
                'validation_failed': True,
                'stage': 'syntax',
                'line': e.lineno
            }), 400
        
        # Step 2: Run validation on RunPod
        if not RUNPOD_API_KEY or not RUNPOD_ENDPOINT_ID:
            return jsonify({
                'error': 'RunPod API key or endpoint ID not configured. Cannot validate kernel remotely.',
                'validation_failed': True,
                'stage': 'configuration'
            }), 500
        
        try:
            print(f"[KERNEL SAVE] Running validation on RunPod before saving...")
            
            runpod_input = {
                'nb_an_projection': 10,
                'nb_scenarios': 2,
                'nb_int_scenarios': 2,
                'max_accounts': 1,
                'kernel_code': content,
                'data_file_urls': {}
            }
            
            endpoint = runpod.Endpoint(RUNPOD_ENDPOINT_ID)
            
            # Use run_sync for synchronous execution with 300s timeout
            try:
                output = endpoint.run_sync(runpod_input, timeout=300)
                print(f"  RunPod validation completed. Output type: {type(output)}")
                
                if isinstance(output, dict) and 'error' in output:
                    error_msg = output['error']
                    tb = output.get('traceback', '')
                    if 'kernel_error' in output:
                        kernel_err = output['kernel_error']
                        return jsonify({
                            'error': kernel_err.get('error', error_msg),
                            'validation_failed': True,
                            'stage': 'kernel_load',
                            'line': kernel_err.get('line'),
                            'traceback': kernel_err.get('traceback', tb)
                        }), 400
                    return jsonify({
                        'error': error_msg,
                        'validation_failed': True,
                        'stage': 'runtime',
                        'traceback': tb
                    }), 400
                    
                # Validation passed!
                print(f"  ✓ Validation passed on RunPod")
                
            except TimeoutError:
                return jsonify({
                    'error': 'Validation timed out after 300 seconds',
                    'validation_failed': True,
                    'stage': 'timeout'
                }), 504
                
        except Exception as e:
            return jsonify({
                'error': f'Validation failed: {str(e)}',
                'validation_failed': True,
                'stage': 'validation',
                'traceback': traceback.format_exc()
            }), 400
    
    version_name = data.get('version_name', f"v_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}")
    description = data.get('description', '')
    
    kernels_path = HERE / 'calculations' / 'kernels.py'
    backup_path = HERE / 'calculations' / 'kernels.py.backup'
    
    try:
        # Backup current file
        if kernels_path.exists():
            backup_path.write_text(kernels_path.read_text())
        
        # Save current version to database before overwriting (mark as not active)
        ph = get_placeholder()
        execute_sql(f"UPDATE kernel_versions SET is_active = 0 WHERE is_active = 1", ())
        
        # Save new version to database
        execute_sql(
            f"""INSERT INTO kernel_versions (version_name, created_at, description, content, is_active)
                VALUES ({ph}, {ph}, {ph}, {ph}, {ph})""",
            (version_name, datetime.utcnow().isoformat(), description, content, 1)
        )
        
        # Write new content to file
        kernels_path.write_text(content)
        
        should_restart = data.get('restart', True)
        
        if should_restart:
            # Schedule restart after response is sent
            import subprocess
            import sys
            
            def restart_server():
                import time
                time.sleep(1)  # Give time for response to be sent
                os.execv(sys.executable, [sys.executable] + sys.argv)
            
            thread = threading.Thread(target=restart_server)
            thread.daemon = True
            thread.start()
            
            return jsonify({
                'success': True,
                'message': 'Validation passed! kernels.py updated, server restarting...',
                'version_name': version_name,
                'backup': str(backup_path),
                'validated': should_validate
            })
        else:
            return jsonify({
                'success': True,
                'message': 'kernels.py updated (restart=false, changes will apply on next server restart)',
                'version_name': version_name,
                'backup': str(backup_path),
                'validated': should_validate
            })
    
    except Exception as e:
        # Restore backup on failure
        if backup_path.exists():
            kernels_path.write_text(backup_path.read_text())
        return jsonify({'error': f'Failed to update kernels.py: {e}'}), 500

@app.route('/admin/kernels/versions', methods=['GET'])
def list_kernel_versions():
    """
    List all saved kernel versions from the database.
    """
    rows = fetch_all("SELECT id, version_name, created_at, description, is_active FROM kernel_versions ORDER BY created_at DESC")
    return jsonify({
        'versions': rows,
        'count': len(rows)
    })

@app.route('/admin/kernels/versions/<int:version_id>', methods=['GET'])
def get_kernel_version(version_id):
    """
    Get a specific kernel version content by ID.
    """
    ph = get_placeholder()
    row = fetch_one(f"SELECT * FROM kernel_versions WHERE id = {ph}", (version_id,))
    if not row:
        return jsonify({'error': 'Version not found'}), 404
    
    return jsonify(row)

@app.route('/admin/kernels/versions/<int:version_id>/activate', methods=['POST'])
def activate_kernel_version(version_id):
    """
    Activate a specific kernel version: write it to kernels.py and restart.
    
    JSON body:
    {
        "password": "admin123"
    }
    """
    password = request.json.get('password') if request.is_json else None
    if password != ADMIN_PASSWORD:
        return jsonify({'error': 'Invalid password'}), 403
    
    ph = get_placeholder()
    row = fetch_one(f"SELECT * FROM kernel_versions WHERE id = {ph}", (version_id,))
    if not row:
        return jsonify({'error': 'Version not found'}), 404
    
    kernels_path = HERE / 'calculations' / 'kernels.py'
    backup_path = HERE / 'calculations' / 'kernels.py.backup'
    
    try:
        # Backup current file
        if kernels_path.exists():
            backup_path.write_text(kernels_path.read_text())
        
        # Update active status in database
        execute_sql(f"UPDATE kernel_versions SET is_active = 0 WHERE is_active = 1", ())
        execute_sql(f"UPDATE kernel_versions SET is_active = 1 WHERE id = {ph}", (version_id,))
        
        # Write content to file
        kernels_path.write_text(row['content'])
        
        # Restart
        import sys
        def restart_server():
            import time
            time.sleep(1)
            os.execv(sys.executable, [sys.executable] + sys.argv)
        
        thread = threading.Thread(target=restart_server)
        thread.daemon = True
        thread.start()
        
        return jsonify({
            'success': True,
            'message': f"Activated version '{row['version_name']}', server restarting...",
            'version_name': row['version_name']
        })
    except Exception as e:
        return jsonify({'error': f'Failed to activate version: {e}'}), 500

@app.route('/admin/kernels/versions/<int:version_id>', methods=['DELETE'])
def delete_kernel_version(version_id):
    """
    Delete a kernel version from the database.
    
    JSON body:
    {
        "password": "admin123"
    }
    """
    password = request.json.get('password') if request.is_json else None
    if password != ADMIN_PASSWORD:
        return jsonify({'error': 'Invalid password'}), 403
    
    ph = get_placeholder()
    row = fetch_one(f"SELECT is_active FROM kernel_versions WHERE id = {ph}", (version_id,))
    if not row:
        return jsonify({'error': 'Version not found'}), 404
    
    if row['is_active']:
        return jsonify({'error': 'Cannot delete the active version'}), 400
    
    execute_sql(f"DELETE FROM kernel_versions WHERE id = {ph}", (version_id,))
    return jsonify({'success': True, 'message': 'Version deleted'})

@app.route('/admin/kernels/restore', methods=['POST'])
def restore_kernel_backup():
    """
    Restore kernels.py from file backup and restart.
    """
    password = request.json.get('password') if request.is_json else None
    if password != ADMIN_PASSWORD:
        return jsonify({'error': 'Invalid password'}), 403
    
    kernels_path = HERE / 'calculations' / 'kernels.py'
    backup_path = HERE / 'calculations' / 'kernels.py.backup'
    
    if not backup_path.exists():
        return jsonify({'error': 'No backup file found'}), 404
    
    try:
        kernels_path.write_text(backup_path.read_text())
        
        # Restart
        import sys
        def restart_server():
            import time
            time.sleep(1)
            os.execv(sys.executable, [sys.executable] + sys.argv)
        
        thread = threading.Thread(target=restart_server)
        thread.daemon = True
        thread.start()
        
        return jsonify({
            'success': True,
            'message': 'Restored from backup, server restarting...'
        })
    except Exception as e:
        return jsonify({'error': f'Failed to restore: {e}'}), 500

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
