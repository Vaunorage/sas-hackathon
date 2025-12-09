#!/usr/bin/env python3
"""
Command-Line Interface for GPU-based Actuarial Projections
Provides terminal-based job management and result retrieval
"""

import os
import sys
import sqlite3
import json
import time
import argparse
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any
import pandas as pd
from tabulate import tabulate

from paths import HERE

# Import the GPU projection function
try:
    from calculations.gpu import run_projection_gpu
    GPU_AVAILABLE = True
except Exception as e:
    print(f"Warning: GPU module not available: {e}")
    GPU_AVAILABLE = False

# =============================================================================
# CONFIGURATION
# =============================================================================

UPLOAD_FOLDER = HERE / 'uploads'
RESULTS_FOLDER = HERE / 'results'
DATABASE = HERE / 'jobs.db'
DEFAULT_DATA_FOLDER = HERE / 'data_in'

# Create directories if they don't exist
UPLOAD_FOLDER.mkdir(exist_ok=True)
RESULTS_FOLDER.mkdir(exist_ok=True)

APP_VERSION = "1.0.0"
ADMIN_PASSWORD = os.getenv('ADMIN_PASSWORD', 'admin123')

# =============================================================================
# DATABASE INITIALIZATION
# =============================================================================

def init_db():
    """Initialize the SQLite database with jobs table"""
    conn = sqlite3.connect(DATABASE)
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
            progress_percent REAL DEFAULT 0.0,
            results_data TEXT
        )
    """)
    
    # Create table for flux_projetes results
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS flux_projetes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
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
    """)
    
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_flux_projetes_job_id 
        ON flux_projetes(job_id)
    """)
    
    # Create table for vp_flux_compte results
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS vp_flux_compte (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
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
    """)
    
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_vp_flux_compte_job_id 
        ON vp_flux_compte(job_id)
    """)
    
    # Create table for vp_flux_total results
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS vp_flux_total (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            job_id TEXT NOT NULL,
            categorie TEXT NOT NULL,
            vp_flux_tot REAL NOT NULL,
            FOREIGN KEY (job_id) REFERENCES jobs(job_id) ON DELETE CASCADE
        )
    """)
    
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_vp_flux_total_job_id 
        ON vp_flux_total(job_id)
    """)
    
    # Migrate existing database to add new columns if they don't exist
    try:
        cursor.execute("SELECT current_batch FROM jobs LIMIT 1")
    except sqlite3.OperationalError:
        cursor.execute("ALTER TABLE jobs ADD COLUMN current_batch INTEGER DEFAULT 0")
        cursor.execute("ALTER TABLE jobs ADD COLUMN total_batches INTEGER DEFAULT 0")
        cursor.execute("ALTER TABLE jobs ADD COLUMN progress_percent REAL DEFAULT 0.0")
    
    # Migrate to add results_data column
    try:
        cursor.execute("SELECT results_data FROM jobs LIMIT 1")
    except sqlite3.OperationalError:
        cursor.execute("ALTER TABLE jobs ADD COLUMN results_data TEXT")
    
    conn.commit()
    conn.close()

# =============================================================================
# DATABASE HELPERS
# =============================================================================

def get_db_connection():
    """Get a database connection"""
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn

def create_job(job_id: str, parameters: Dict[str, Any], uploaded_files: list) -> None:
    """Create a new job in the database"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        INSERT INTO jobs (job_id, status, created_at, parameters, uploaded_files)
        VALUES (?, ?, ?, ?, ?)
    """, (
        job_id,
        'pending',
        datetime.utcnow().isoformat(),
        json.dumps(parameters),
        json.dumps(uploaded_files)
    ))
    
    conn.commit()
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
    
    set_clause = ', '.join([f"{k} = ?" for k in updates.keys()])
    values = list(updates.values()) + [job_id]
    
    cursor.execute(f"UPDATE jobs SET {set_clause} WHERE job_id = ?", values)
    conn.commit()
    conn.close()

def update_job_progress(job_id: str, current_batch: int, total_batches: int) -> None:
    """Update job progress"""
    progress_percent = (current_batch / total_batches * 100.0) if total_batches > 0 else 0.0
    
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        UPDATE jobs 
        SET current_batch = ?, total_batches = ?, progress_percent = ?
        WHERE job_id = ?
    """, (current_batch, total_batches, progress_percent, job_id))
    
    conn.commit()
    conn.close()

def update_job_results(job_id: str, result_files: list) -> None:
    """Update job with result files"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        UPDATE jobs SET result_files = ? WHERE job_id = ?
    """, (json.dumps(result_files), job_id))
    
    conn.commit()
    conn.close()

def update_job_results_data(job_id: str, results_data: dict) -> None:
    """Update job with results data from run_projection_gpu (legacy JSON storage)"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        UPDATE jobs SET results_data = ? WHERE job_id = ?
    """, (json.dumps(results_data), job_id))
    
    conn.commit()
    conn.close()

def save_flux_projetes(job_id: str, df: pd.DataFrame) -> None:
    """Save flux_projetes DataFrame to database table"""
    conn = get_db_connection()
    
    # Prepare data for bulk insert
    df_copy = df.copy()
    df_copy.insert(0, 'job_id', job_id)
    
    # Rename columns to match database schema (lowercase)
    df_copy.columns = df_copy.columns.str.lower()
    
    # Insert into database
    df_copy.to_sql('flux_projetes', conn, if_exists='append', index=False)
    conn.close()
    print(f"  Saved {len(df)} flux_projetes records to database")

def save_vp_flux_compte(job_id: str, df: pd.DataFrame) -> None:
    """Save vp_flux_compte DataFrame to database table"""
    conn = get_db_connection()
    
    # Prepare data for bulk insert
    df_copy = df.copy()
    df_copy.insert(0, 'job_id', job_id)
    
    # Rename columns to match database schema (lowercase)
    df_copy.columns = df_copy.columns.str.lower()
    
    # Insert into database
    df_copy.to_sql('vp_flux_compte', conn, if_exists='append', index=False)
    conn.close()
    print(f"  Saved {len(df)} vp_flux_compte records to database")

def save_vp_flux_total(job_id: str, df: pd.DataFrame) -> None:
    """Save vp_flux_total DataFrame to database table"""
    conn = get_db_connection()
    
    # Prepare data for bulk insert
    df_copy = df.copy()
    df_copy.insert(0, 'job_id', job_id)
    
    # Rename columns to match database schema (lowercase)
    df_copy.columns = df_copy.columns.str.lower()
    
    # Insert into database
    df_copy.to_sql('vp_flux_total', conn, if_exists='append', index=False)
    conn.close()
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
        query = "SELECT * FROM flux_projetes WHERE job_id = ?"
        params = [job_id]
        
        if an_eval is not None:
            query += " AND an_eval = ?"
            params.append(an_eval)
        
        if mois_eval is not None:
            query += " AND mois_eval = ?"
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
        query = "SELECT * FROM vp_flux_compte WHERE job_id = ?"
        params = [job_id]
        
        if id_compte is not None:
            query += " AND id_compte = ?"
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
        df = pd.read_sql_query(
            "SELECT * FROM vp_flux_total WHERE job_id = ?",
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
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,))
    row = cursor.fetchone()
    conn.close()
    
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
        
        try:
            job_data['current_batch'] = row['current_batch'] if row['current_batch'] is not None else 0
            job_data['total_batches'] = row['total_batches'] if row['total_batches'] is not None else 0
            job_data['progress_percent'] = row['progress_percent'] if row['progress_percent'] is not None else 0.0
        except (KeyError, IndexError):
            job_data['current_batch'] = 0
            job_data['total_batches'] = 0
            job_data['progress_percent'] = 0.0
        
        # Add results_data field
        try:
            job_data['results_data'] = json.loads(row['results_data']) if row['results_data'] else None
        except (KeyError, IndexError):
            job_data['results_data'] = None
        
        return job_data
    return None

def get_all_jobs() -> list:
    """Get all jobs ordered by creation date"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("SELECT * FROM jobs ORDER BY created_at DESC")
    rows = cursor.fetchall()
    conn.close()
    
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
        
        try:
            job_data['current_batch'] = row['current_batch'] if row['current_batch'] is not None else 0
            job_data['total_batches'] = row['total_batches'] if row['total_batches'] is not None else 0
            job_data['progress_percent'] = row['progress_percent'] if row['progress_percent'] is not None else 0.0
        except (KeyError, IndexError):
            job_data['current_batch'] = 0
            job_data['total_batches'] = 0
            job_data['progress_percent'] = 0.0
        
        # Add results_data field
        try:
            job_data['results_data'] = json.loads(row['results_data']) if row['results_data'] else None
        except (KeyError, IndexError):
            job_data['results_data'] = None
        
        jobs.append(job_data)
    return jobs

# =============================================================================
# JOB PROCESSING
# =============================================================================

def get_job_results_folder(job_id: str) -> Path:
    """Get results folder for a specific job"""
    folder = RESULTS_FOLDER / job_id
    folder.mkdir(exist_ok=True)
    return folder

def process_job(job_id: str):
    """Process a job synchronously"""
    try:
        update_job_status(job_id, 'running')
        
        job = get_job(job_id)
        if not job:
            raise Exception(f"Job {job_id} not found")
        
        params = job['parameters']
        nb_years = params.get('nb_an_projection', 100)
        nb_scenarios = params.get('nb_scenarios', 100)
        max_accounts = params.get('max_accounts', None)
        debug_account = params.get('debug_account', None)
        
        data_path = DEFAULT_DATA_FOLDER
        output_path = get_job_results_folder(job_id)
        
        def progress_callback(current_batch: int, total_batches: int):
            """Callback to update job progress"""
            update_job_progress(job_id, current_batch, total_batches)
            progress_percent = (current_batch / total_batches * 100.0) if total_batches > 0 else 0.0
            print(f"  Progress: {current_batch}/{total_batches} batches ({progress_percent:.1f}%)")
        
        print(f"\n{'='*60}")
        print(f"Running GPU projection for job {job_id}")
        print(f"{'='*60}")
        print(f"  Years: {nb_years}")
        print(f"  Scenarios: {nb_scenarios}")
        if max_accounts:
            print(f"  Max Accounts: {max_accounts}")
        if debug_account:
            print(f"  Debug Account: {debug_account}")
        print(f"{'='*60}\n")
        
        results = run_projection_gpu(
            data_path=data_path,
            output_path=output_path,
            nb_an_projection=nb_years,
            nb_scenarios=nb_scenarios,
            max_accounts=max_accounts,
            debug_account=debug_account,
            progress_callback=progress_callback
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
            output_file_metadata = {
                'FLUX_PROJETES_GPU.csv': {
                    'type': 'internal',
                    'description': 'Projected cash flows by time period'
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
                
                if file.name in output_file_metadata:
                    file_info.update(output_file_metadata[file.name])
                elif file.name.startswith('DEBUG_account_'):
                    file_info['type'] = 'debug'
                    file_info['description'] = 'Debug trace for specific account/scenario'
                else:
                    file_info['type'] = 'other'
                    file_info['description'] = 'Additional output file'
                
                result_files.append(file_info)
        
        update_job_results(job_id, result_files)
        update_job_status(job_id, 'completed')
        
        print(f"\n{'='*60}")
        print(f"✓ Job {job_id} completed successfully")
        print(f"{'='*60}")
        print(f"  Saved {len(result_files)} output files:")
        for file_info in result_files:
            size_mb = file_info['size'] / (1024 * 1024)
            print(f"    - {file_info['name']} ({file_info['type']}, {size_mb:.2f} MB)")
            print(f"      {file_info['description']}")
        print(f"{'='*60}\n")
        
        return True
        
    except Exception as e:
        import traceback
        error_msg = f"{str(e)}\n{traceback.format_exc()}"
        print(f"\n✗ Job {job_id} failed: {error_msg}")
        update_job_status(job_id, 'failed', error_message=error_msg)
        return False

# =============================================================================
# CLI COMMANDS
# =============================================================================

def cmd_run(args):
    """Run a new projection job"""
    if not GPU_AVAILABLE:
        print("✗ Error: GPU module not available")
        return 1
    
    # Generate job ID
    job_id = f"job_{datetime.utcnow().strftime('%Y%m%d_%H%M%S_%f')}"
    
    # Build parameters
    parameters = {
        'nb_an_projection': args.years,
        'nb_scenarios': args.scenarios,
        'max_accounts': args.max_accounts,
        'debug_account': args.debug_account
    }
    
    print(f"Creating job: {job_id}")
    create_job(job_id, parameters, [])
    
    if args.async_mode:
        print(f"✓ Job created: {job_id}")
        print(f"  Use 'cli.py status {job_id}' to check progress")
        print(f"  Use 'cli.py watch {job_id}' to monitor in real-time")
        return 0
    else:
        # Run synchronously
        success = process_job(job_id)
        return 0 if success else 1

def cmd_status(args):
    """Check status of a job"""
    job = get_job(args.job_id)
    
    if not job:
        print(f"✗ Job not found: {args.job_id}")
        return 1
    
    print(f"\n{'='*60}")
    print(f"Job: {job['job_id']}")
    print(f"{'='*60}")
    print(f"Status: {job['status'].upper()}")
    print(f"Created: {job['created_at']}")
    if job['started_at']:
        print(f"Started: {job['started_at']}")
    if job['completed_at']:
        print(f"Completed: {job['completed_at']}")
    
    if job['status'] == 'running':
        print(f"\nProgress:")
        print(f"  Batch: {job['current_batch']}/{job['total_batches']}")
        print(f"  Percent: {job['progress_percent']:.1f}%")
    
    print(f"\nParameters:")
    params = job['parameters']
    print(f"  Years: {params.get('nb_an_projection', 'N/A')}")
    print(f"  Scenarios: {params.get('nb_scenarios', 'N/A')}")
    if params.get('max_accounts'):
        print(f"  Max Accounts: {params.get('max_accounts')}")
    if params.get('debug_account'):
        print(f"  Debug Account: {params.get('debug_account')}")
    
    if job['error_message']:
        print(f"\nError:")
        print(f"  {job['error_message']}")
    
    if job['result_files']:
        print(f"\nResult Files ({len(job['result_files'])}):")
        for file_info in job['result_files']:
            size_mb = file_info['size'] / (1024 * 1024)
            print(f"  - {file_info['name']} ({size_mb:.2f} MB)")
            if 'description' in file_info:
                print(f"    {file_info['description']}")
    
    print(f"{'='*60}\n")
    return 0

def cmd_watch(args):
    """Watch a job in real-time"""
    print(f"Monitoring job: {args.job_id}")
    print("Press Ctrl+C to stop watching\n")
    
    try:
        last_status = None
        last_progress = -1
        
        while True:
            job = get_job(args.job_id)
            
            if not job:
                print(f"✗ Job not found: {args.job_id}")
                return 1
            
            status = job['status']
            progress = job['progress_percent']
            
            if status != last_status:
                print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Status changed: {status.upper()}")
                last_status = status
            
            if status == 'running' and progress != last_progress:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Progress: {job['current_batch']}/{job['total_batches']} ({progress:.1f}%)")
                last_progress = progress
            
            if status in ['completed', 'failed', 'cancelled']:
                print(f"\n{'='*60}")
                if status == 'completed':
                    print(f"✓ Job completed successfully")
                    if job['result_files']:
                        print(f"\nResult files:")
                        for file_info in job['result_files']:
                            print(f"  - {file_info['name']}")
                elif status == 'failed':
                    print(f"✗ Job failed")
                    if job['error_message']:
                        print(f"\nError: {job['error_message']}")
                else:
                    print(f"⊗ Job cancelled")
                print(f"{'='*60}\n")
                break
            
            time.sleep(args.interval)
            
    except KeyboardInterrupt:
        print("\n\nStopped watching.")
        return 0
    
    return 0

def cmd_list(args):
    """List all jobs"""
    jobs = get_all_jobs()
    
    if not jobs:
        print("No jobs found.")
        return 0
    
    # Filter by status if specified
    if args.status:
        jobs = [j for j in jobs if j['status'] == args.status]
        if not jobs:
            print(f"No jobs with status: {args.status}")
            return 0
    
    # Prepare table data
    table_data = []
    for job in jobs[:args.limit] if args.limit else jobs:
        row = [
            job['job_id'][:20],
            job['status'].upper(),
            job['created_at'][:19] if job['created_at'] else 'N/A',
            f"{job['progress_percent']:.1f}%" if job['status'] == 'running' else '-',
            len(job['result_files']) if job['result_files'] else 0
        ]
        table_data.append(row)
    
    headers = ['Job ID', 'Status', 'Created', 'Progress', 'Files']
    print(f"\nTotal jobs: {len(jobs)}\n")
    print(tabulate(table_data, headers=headers, tablefmt='grid'))
    print()
    
    return 0

def cmd_results(args):
    """View job results from database or CSV files"""
    job = get_job(args.job_id)
    
    if not job:
        print(f"✗ Job not found: {args.job_id}")
        return 1
    
    if job['status'] != 'completed':
        print(f"✗ Job not completed (status: {job['status']})")
        return 1
    
    try:
        # Retrieve from database (preferred) or fallback to CSV
        source = "database"
        filter_info = []
        
        if args.type == 'summary':
            df = get_vp_flux_total(args.job_id)
            result_name = "VP_FLUX_TOTAL"
        elif args.type == 'detailed':
            # Apply ID_COMPTE filter if specified
            id_compte = getattr(args, 'id_compte', None)
            df = get_vp_flux_compte(args.job_id, id_compte=id_compte)
            result_name = "VP_FLUX_COMPTE"
            if id_compte is not None:
                filter_info.append(f"ID_COMPTE={id_compte}")
        elif args.type == 'internal':
            # Apply AN_EVAL and MOIS_EVAL filters if specified
            an_eval = getattr(args, 'an_eval', None)
            mois_eval = getattr(args, 'mois_eval', None)
            df = get_flux_projetes(args.job_id, an_eval=an_eval, mois_eval=mois_eval)
            result_name = "FLUX_PROJETES"
            if an_eval is not None:
                filter_info.append(f"AN_EVAL={an_eval}")
            if mois_eval is not None:
                filter_info.append(f"MOIS_EVAL={mois_eval}")
        else:
            print(f"✗ Invalid result type: {args.type}")
            print(f"  Valid types: summary, detailed, internal")
            return 1
        
        # Fallback to CSV if database query returns no data
        if df is None or len(df) == 0:
            source = "CSV file"
            result_file_map = {
                'summary': 'VP_FLUX_TOTAL_GPU.csv',
                'detailed': 'VP_FLUX_COMPTE_GPU.csv',
                'internal': 'FLUX_PROJETES_GPU.csv'
            }
            results_folder = get_job_results_folder(args.job_id)
            file_path = results_folder / result_file_map[args.type]
            
            if not file_path.exists():
                print(f"✗ Results not found in database or CSV files")
                return 1
            
            df = pd.read_csv(file_path, sep=';')
        
        total_rows = len(df)
        
        # Apply row limit if specified
        display_limit = args.limit if args.limit else 10
        if args.limit and args.limit < total_rows:
            df_display = df.head(args.limit)
        else:
            df_display = df
        
        print(f"\n{'='*60}")
        print(f"Results: {result_name} (source: {source})")
        if filter_info:
            print(f"Filters: {', '.join(filter_info)}")
        print(f"{'='*60}")
        print(f"Total Rows: {total_rows}")
        print(f"Displaying: {len(df_display)} rows")
        print(f"Columns: {len(df.columns)}")
        print(f"{'='*60}\n")
        
        if args.format == 'table':
            print(tabulate(df_display, headers='keys', tablefmt='grid', showindex=False))
        elif args.format == 'csv':
            print(df_display.to_csv(sep=';', index=False))
        elif args.format == 'json':
            print(df_display.to_json(orient='records', indent=2))
        
        print()
        
        if args.save:
            output_file = Path(args.save)
            if args.format == 'csv':
                df.to_csv(output_file, sep=';', index=False)
            elif args.format == 'json':
                df.to_json(output_file, orient='records', indent=2)
            print(f"✓ Saved {total_rows} rows to: {output_file}")
        
        return 0
        
    except Exception as e:
        print(f"✗ Error reading results: {e}")
        import traceback
        traceback.print_exc()
        return 1

def cmd_get_all_results(args):
    """Retrieve all result DataFrames from database"""
    job = get_job(args.job_id)
    
    if not job:
        print(f"✗ Job not found: {args.job_id}")
        return 1
    
    if job['status'] != 'completed':
        print(f"✗ Job not completed (status: {job['status']})")
        return 1
    
    try:
        print(f"\n{'='*60}")
        print(f"Retrieving all results for job: {args.job_id}")
        print(f"{'='*60}\n")
        
        # Retrieve all three result types
        flux_projetes_df = get_flux_projetes(args.job_id)
        vp_flux_compte_df = get_vp_flux_compte(args.job_id)
        vp_flux_total_df = get_vp_flux_total(args.job_id)
        
        results = {
            'flux_projetes': flux_projetes_df,
            'vp_flux_compte': vp_flux_compte_df,
            'vp_flux_total': vp_flux_total_df
        }
        
        limit = args.limit if args.limit else 10
        
        for name, df in results.items():
            print(f"\n{name.upper()}:")
            print(f"{'-'*60}")
            if df is not None and len(df) > 0:
                print(f"Total rows: {len(df)}")
                print(f"Columns: {list(df.columns)}\n")
                
                df_display = df.head(limit) if limit < len(df) else df
                print(tabulate(df_display, headers='keys', tablefmt='grid', showindex=False))
                
                if len(df) > limit:
                    print(f"\n(Showing first {limit} of {len(df)} rows)")
            else:
                print("No data available")
            print()
        
        # Show summary
        print(f"\n{'='*60}")
        print("SUMMARY:")
        print(f"  flux_projetes: {len(flux_projetes_df) if flux_projetes_df is not None else 0} rows")
        print(f"  vp_flux_compte: {len(vp_flux_compte_df) if vp_flux_compte_df is not None else 0} rows")
        print(f"  vp_flux_total: {len(vp_flux_total_df) if vp_flux_total_df is not None else 0} rows")
        print(f"{'='*60}\n")
        
        return 0
        
    except Exception as e:
        print(f"✗ Error retrieving results: {e}")
        import traceback
        traceback.print_exc()
        return 1

def cmd_clear(args):
    """Clear database"""
    if not args.confirm:
        print("✗ This will delete all jobs from the database!")
        print("  Use --confirm to proceed")
        return 1
    
    password = args.password or input("Enter admin password: ")
    
    if password != ADMIN_PASSWORD:
        print("✗ Invalid password")
        return 1
    
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM jobs")
        deleted_count = cursor.rowcount
        conn.commit()
        conn.close()
        
        if args.delete_files:
            import shutil
            for folder in [UPLOAD_FOLDER, RESULTS_FOLDER]:
                if folder.exists():
                    for item in folder.iterdir():
                        if item.is_dir():
                            shutil.rmtree(item)
                        else:
                            item.unlink()
            print(f"✓ Deleted {deleted_count} jobs and all associated files")
        else:
            print(f"✓ Deleted {deleted_count} jobs from database")
        
        return 0
        
    except Exception as e:
        print(f"✗ Error clearing database: {e}")
        return 1

def cmd_info(args):
    """Show system information"""
    print(f"\n{'='*60}")
    print(f"GPU Actuarial Projections CLI")
    print(f"{'='*60}")
    print(f"Version: {APP_VERSION}")
    print(f"GPU Available: {GPU_AVAILABLE}")
    print(f"Database: {DATABASE}")
    print(f"Default Data: {DEFAULT_DATA_FOLDER}")
    print(f"Results Folder: {RESULTS_FOLDER}")
    
    # Count jobs by status
    jobs = get_all_jobs()
    status_counts = {}
    for job in jobs:
        status = job['status']
        status_counts[status] = status_counts.get(status, 0) + 1
    
    print(f"\nJobs:")
    print(f"  Total: {len(jobs)}")
    for status, count in sorted(status_counts.items()):
        print(f"  {status.capitalize()}: {count}")
    
    print(f"{'='*60}\n")
    return 0

# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main CLI entry point"""
    # Initialize database
    init_db()
    
    parser = argparse.ArgumentParser(
        description='GPU Actuarial Projections CLI',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Run command
    run_parser = subparsers.add_parser('run', help='Run a new projection job')
    run_parser.add_argument('--years', type=int, default=100, help='Number of years to project (default: 100)')
    run_parser.add_argument('--scenarios', type=int, default=100, help='Number of scenarios (default: 100)')
    run_parser.add_argument('--max-accounts', type=int, help='Maximum number of accounts to process')
    run_parser.add_argument('--debug-account', type=int, help='Account ID for debugging')
    run_parser.add_argument('--async', dest='async_mode', action='store_true', help='Run asynchronously (return immediately)')
    
    # Status command
    status_parser = subparsers.add_parser('status', help='Check status of a job')
    status_parser.add_argument('job_id', help='Job ID to check')
    
    # Watch command
    watch_parser = subparsers.add_parser('watch', help='Watch a job in real-time')
    watch_parser.add_argument('job_id', help='Job ID to watch')
    watch_parser.add_argument('--interval', type=float, default=2.0, help='Update interval in seconds (default: 2.0)')
    
    # List command
    list_parser = subparsers.add_parser('list', help='List all jobs')
    list_parser.add_argument('--status', help='Filter by status')
    list_parser.add_argument('--limit', type=int, help='Limit number of jobs shown')
    
    # Results command
    results_parser = subparsers.add_parser('results', help='View job results')
    results_parser.add_argument('job_id', help='Job ID')
    results_parser.add_argument('--type', default='summary', choices=['summary', 'detailed', 'internal'], help='Result type (default: summary)')
    results_parser.add_argument('--format', default='table', choices=['table', 'csv', 'json'], help='Output format (default: table)')
    results_parser.add_argument('--limit', type=int, help='Limit number of rows')
    results_parser.add_argument('--save', help='Save results to file')
    # Filters for flux_projetes (internal)
    results_parser.add_argument('--an-eval', type=int, dest='an_eval', help='Filter by year (for internal type)')
    results_parser.add_argument('--mois-eval', type=int, dest='mois_eval', help='Filter by month (for internal type)')
    # Filter for vp_flux_compte (detailed)
    results_parser.add_argument('--id-compte', type=int, dest='id_compte', help='Filter by account ID (for detailed type)')
    
    # Get all results command
    get_all_parser = subparsers.add_parser('get-all-results', help='Retrieve all result types from database')
    get_all_parser.add_argument('job_id', help='Job ID')
    get_all_parser.add_argument('--limit', type=int, default=10, help='Limit number of rows displayed per table (default: 10)')
    
    # Clear command
    clear_parser = subparsers.add_parser('clear', help='Clear all jobs from database')
    clear_parser.add_argument('--confirm', action='store_true', help='Confirm deletion')
    clear_parser.add_argument('--delete-files', action='store_true', help='Also delete uploaded and result files')
    clear_parser.add_argument('--password', help='Admin password')
    
    # Info command
    info_parser = subparsers.add_parser('info', help='Show system information')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 0
    
    # Route to appropriate command handler
    commands = {
        'run': cmd_run,
        'status': cmd_status,
        'watch': cmd_watch,
        'list': cmd_list,
        'results': cmd_results,
        'get-all-results': cmd_get_all_results,
        'clear': cmd_clear,
        'info': cmd_info
    }
    
    handler = commands.get(args.command)
    if handler:
        return handler(args)
    else:
        print(f"Unknown command: {args.command}")
        return 1

if __name__ == '__main__':
    sys.exit(main())
