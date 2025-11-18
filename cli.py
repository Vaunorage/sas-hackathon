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
    from gpu import run_projection_gpu
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
            progress_percent REAL DEFAULT 0.0
        )
    """)
    
    # Migrate existing database to add new columns if they don't exist
    try:
        cursor.execute("SELECT current_batch FROM jobs LIMIT 1")
    except sqlite3.OperationalError:
        cursor.execute("ALTER TABLE jobs ADD COLUMN current_batch INTEGER DEFAULT 0")
        cursor.execute("ALTER TABLE jobs ADD COLUMN total_batches INTEGER DEFAULT 0")
        cursor.execute("ALTER TABLE jobs ADD COLUMN progress_percent REAL DEFAULT 0.0")
    
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
    """View job results"""
    job = get_job(args.job_id)
    
    if not job:
        print(f"✗ Job not found: {args.job_id}")
        return 1
    
    if job['status'] != 'completed':
        print(f"✗ Job not completed (status: {job['status']})")
        return 1
    
    # Map result types to files
    result_file_map = {
        'summary': 'VP_FLUX_TOTAL_GPU.csv',
        'detailed': 'VP_FLUX_COMPTE_GPU.csv',
        'internal': 'FLUX_PROJETES_GPU.csv'
    }
    
    result_file = result_file_map.get(args.type)
    if not result_file:
        print(f"✗ Invalid result type: {args.type}")
        print(f"  Valid types: {', '.join(result_file_map.keys())}")
        return 1
    
    # Read the result file
    results_folder = get_job_results_folder(args.job_id)
    file_path = results_folder / result_file
    
    if not file_path.exists():
        print(f"✗ Result file not found: {result_file}")
        return 1
    
    try:
        df = pd.read_csv(file_path, sep=';', nrows=args.limit if args.limit else None)
        
        print(f"\n{'='*60}")
        print(f"Results: {result_file}")
        print(f"{'='*60}")
        print(f"Rows: {len(df)}")
        print(f"Columns: {len(df.columns)}")
        print(f"{'='*60}\n")
        
        if args.format == 'table':
            print(tabulate(df.head(args.limit if args.limit else 10), headers='keys', tablefmt='grid'))
        elif args.format == 'csv':
            print(df.to_csv(sep=';', index=False))
        elif args.format == 'json':
            print(df.to_json(orient='records', indent=2))
        
        print()
        
        if args.save:
            output_file = Path(args.save)
            if args.format == 'csv':
                df.to_csv(output_file, sep=';', index=False)
            elif args.format == 'json':
                df.to_json(output_file, orient='records', indent=2)
            print(f"✓ Saved to: {output_file}")
        
        return 0
        
    except Exception as e:
        print(f"✗ Error reading results: {e}")
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
