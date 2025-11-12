import os
import logging
import json
import threading
from datetime import datetime
from flask import Flask, request, jsonify, Response
from flask_cors import CORS
from numba import cuda
from werkzeug.exceptions import HTTPException
import pandas as pd
import io

# Import your ACFC algorithm
from algo1.gpu1 import gpu_acfc_algorithm_complete, initialize_cuda_for_thread, _thread_local

from paths import HERE

### NEW: DATABASE CONFIGURATION ###
# Set this to True to use NeonDB (PostgreSQL), False to use SQLite
USE_NEONDB = True

# Conditionally import libraries and load environment variables for NeonDB
if USE_NEONDB:
    try:
        import psycopg
        from dotenv import load_dotenv

        load_dotenv()  # Load environment variables from .env file
    except ImportError:
        raise ImportError(
            "psycopg and python-dotenv are required to use NeonDB. Please run 'pip install \"psycopg[binary]\" python-dotenv'")
else:
    import sqlite3

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

INITIALIZED = threading.Event()
INITIALIZATION_ERROR = None

# Enable CORS for all routes
# CORS(app,
#      resources={r"/*": {"origins": "*"}},
#      allow_headers=["Content-Type", "Authorization"],
#      methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"]
# )

# Configuration
PORT = int(os.environ.get('PORT', 80))
ENV = os.environ.get('ENVIRONMENT', 'production')
DATA_PATH = HERE.joinpath("data_in")

# Database connection details
DB_PATH = HERE.joinpath('app/jobs.db')  # Used only if USE_NEONDB is False
NEON_DB_URL = os.getenv("DATABASE_URL")  # Used only if USE_NEONDB is True


### NEW: Database connection and SQL helper functions ###
def get_db_connection():
    """Returns a database connection object based on the USE_NEONDB flag."""
    if USE_NEONDB:
        if not NEON_DB_URL:
            raise ValueError("DATABASE_URL environment variable is not set. Cannot connect to NeonDB.")
        try:
            conn = psycopg.connect(NEON_DB_URL)
            return conn
        except Exception as e:
            logger.error(f"Failed to connect to NeonDB: {e}")
            raise
    else:
        # Ensure the directory for the SQLite DB exists
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        return sqlite3.connect(DB_PATH)


def get_placeholder():
    """Returns the correct SQL parameter placeholder based on the database type."""
    return "%s" if USE_NEONDB else "?"


### MODIFIED: Database initialization ###
def init_db():
    """Initialize database tables if they don't exist, for either SQLite or NeonDB."""
    logger.info(f"Initializing database ({'NeonDB' if USE_NEONDB else 'SQLite'})...")

    if USE_NEONDB:
        # PostgreSQL syntax with SERIAL for auto-incrementing keys
        create_jobs_table = '''
                            CREATE TABLE IF NOT EXISTS jobs \
                            ( \
                                job_id \
                                TEXT \
                                PRIMARY \
                                KEY, \
                                status \
                                TEXT, \
                                created_at \
                                TEXT, \
                                started_at \
                                TEXT, \
                                completed_at \
                                TEXT, \
                                error \
                                TEXT, \
                                parameters \
                                TEXT
                            );'''
        create_results_table = '''
                               CREATE TABLE IF NOT EXISTS results \
                               ( \
                                   id \
                                   SERIAL \
                                   PRIMARY \
                                   KEY, \
                                   job_id \
                                   TEXT, \
                                   result_type \
                                   TEXT, \
                                   data \
                                   TEXT, \
                                   FOREIGN \
                                   KEY \
                               ( \
                                   job_id \
                               ) REFERENCES jobs \
                               ( \
                                   job_id \
                               )
                                   );'''
        create_job_files_table = '''
                                 CREATE TABLE IF NOT EXISTS job_files \
                                 ( \
                                     file_id \
                                     SERIAL \
                                     PRIMARY \
                                     KEY, \
                                     job_id \
                                     TEXT, \
                                     file_name \
                                     TEXT, \
                                     file_content \
                                     TEXT, \
                                     FOREIGN \
                                     KEY \
                                 ( \
                                     job_id \
                                 ) REFERENCES jobs \
                                 ( \
                                     job_id \
                                 )
                                     );'''
    else:
        # SQLite syntax with INTEGER PRIMARY KEY AUTOINCREMENT
        create_jobs_table = '''
                            CREATE TABLE IF NOT EXISTS jobs \
                            ( \
                                job_id \
                                TEXT \
                                PRIMARY \
                                KEY, \
                                status \
                                TEXT, \
                                created_at \
                                TEXT, \
                                started_at \
                                TEXT, \
                                completed_at \
                                TEXT, \
                                error \
                                TEXT, \
                                parameters \
                                TEXT
                            );'''
        create_results_table = '''
                               CREATE TABLE IF NOT EXISTS results \
                               ( \
                                   id \
                                   INTEGER \
                                   PRIMARY \
                                   KEY \
                                   AUTOINCREMENT, \
                                   job_id \
                                   TEXT, \
                                   result_type \
                                   TEXT, \
                                   data \
                                   TEXT, \
                                   FOREIGN \
                                   KEY \
                               ( \
                                   job_id \
                               ) REFERENCES jobs \
                               ( \
                                   job_id \
                               )
                                   );'''
        create_job_files_table = '''
                                 CREATE TABLE IF NOT EXISTS job_files \
                                 ( \
                                     file_id \
                                     INTEGER \
                                     PRIMARY \
                                     KEY \
                                     AUTOINCREMENT, \
                                     job_id \
                                     TEXT, \
                                     file_name \
                                     TEXT, \
                                     file_content \
                                     TEXT, \
                                     FOREIGN \
                                     KEY \
                                 ( \
                                     job_id \
                                 ) REFERENCES jobs \
                                 ( \
                                     job_id \
                                 )
                                     );'''

    try:
        # Use a 'with' statement for automatic commit/close
        with get_db_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(create_jobs_table)
                cursor.execute(create_results_table)
                cursor.execute(create_job_files_table)
        logger.info("Database initialized successfully.")
    except Exception as e:
        logger.error(f"Database initialization failed: {e}", exc_info=True)
        global INITIALIZATION_ERROR
        INITIALIZATION_ERROR = f"Database initialization failed: {e}"
        raise


# Initialize DB at startup and set readiness flags
try:
    init_db()
    INITIALIZED.set()
except Exception:
    # Error is already logged and stored in init_db
    pass


@app.route('/ping')
def ping():
    return jsonify({"status": "healthy"}), 200


@app.route('/ready')
def ready():
    if INITIALIZED.is_set() and INITIALIZATION_ERROR is None:
        return jsonify({"status": "ready"}), 200
    elif INITIALIZATION_ERROR:
        return jsonify({"status": "error", "error": INITIALIZATION_ERROR}), 503
    else:
        return jsonify({"status": "initializing"}), 503


### MODIFIED: Job execution function ###
def run_job(job_id, params, dataframes):
    ph = get_placeholder()  # Get correct placeholder (%s or ?)

    try:
        # Connect once and perform all operations

        initialize_cuda_for_thread()

        with get_db_connection() as conn:
            # Update status to running
            with conn.cursor() as cursor:
                sql_update_status = f"UPDATE jobs SET status = {ph}, started_at = {ph} WHERE job_id = {ph}"
                cursor.execute(sql_update_status, ('running', datetime.utcnow().isoformat(), job_id))

            # Run the algorithm (no changes here)
            results, detailed_results, internal_results = gpu_acfc_algorithm_complete(
                data_path=DATA_PATH, nb_accounts=params.get('nb_accounts', 4),
                nb_scenarios=params.get('nb_scenarios', 10), nb_years=params.get('nb_years', 10),
                nb_sc_int=params.get('nb_sc_int', 10), nb_an_projection_int=params.get('nb_an_projection_int', 10),
                choc_capital=params.get('choc_capital', 0.35), hurdle_rt=params.get('hurdle_rt', 0.10),
                log_account_id=params.get('log_account_id'), log_scenario=params.get('log_scenario'),
                log_max_years=params.get('log_max_years'), log_internal_scenario=params.get('log_internal_scenario'),
                **dataframes
            )

            # Store results and update status to completed
            with conn.cursor() as cursor:
                sql_insert_results = f"INSERT INTO results (job_id, result_type, data) VALUES ({ph}, {ph}, {ph})"
                cursor.execute(sql_insert_results, (job_id, 'summary', results.to_json(orient='records')))
                cursor.execute(sql_insert_results, (job_id, 'detailed', detailed_results.to_json(orient='records')))
                cursor.execute(sql_insert_results, (job_id, 'internal', internal_results.to_json(orient='records')))

                sql_update_completed = f"UPDATE jobs SET status = {ph}, completed_at = {ph} WHERE job_id = {ph}"
                cursor.execute(sql_update_completed, ('completed', datetime.utcnow().isoformat(), job_id))
            # For both psycopg and sqlite3, the 'with conn' block handles the transaction commit.

    except Exception as e:
        logger.error(f"Job {job_id} failed: {e}", exc_info=True)
        try:
            # Attempt to update job status to 'failed' in the DB
            with get_db_connection() as conn:
                with conn.cursor() as cursor:
                    sql_update_failed = f"UPDATE jobs SET status = {ph}, completed_at = {ph}, error = {ph} WHERE job_id = {ph}"
                    cursor.execute(sql_update_failed, ('failed', datetime.utcnow().isoformat(), str(e), job_id))
        except Exception as db_err:
            logger.error(f"CRITICAL: Could not update job {job_id} status to failed: {db_err}")

    finally:
        if cuda.is_available() and getattr(_thread_local, 'cuda_initialized', False):
            logger.info(f"Closing CUDA context for thread: {threading.get_ident()}")
            cuda.close()
            _thread_local.cuda_initialized = False

# Error handlers (no changes)
@app.errorhandler(404)
def not_found(e): return jsonify({"error": "Resource not found"}), 404


@app.errorhandler(500)
def internal_error(e):
    logger.error(f"Internal server error: {str(e)}")
    return jsonify({"error": "Internal server error"}), 500


@app.errorhandler(Exception)
def handle_exception(e):
    if isinstance(e, HTTPException): return jsonify({"error": e.description}), e.code
    logger.error(f"Unhandled exception: {str(e)}", exc_info=True)
    return jsonify({"error": "An unexpected error occurred"}), 500


@app.before_request
def log_request():
    logger.info(f"{request.method} {request.path} - {request.remote_addr}")


# Routes
@app.route('/')
def hello():
    return jsonify({"message": "ACFC Algorithm API", "version": "1.0.0", "environment": ENV})


### MODIFIED: create_job route ###
@app.route('/jobs', methods=['POST'])
def create_job():
    job_id = f"job_{datetime.utcnow().strftime('%Y%m%d_%H%M%S_%f')}"
    params = {}
    dataframes = {}
    ph = get_placeholder()

    try:
        with get_db_connection() as conn:
            with conn.cursor() as cursor:
                # Handle file uploads
                if request.files:
                    for key, file in request.files.items():
                        if file and file.filename:
                            content = file.stream.read().decode('utf-8')
                            sql_insert_file = f"INSERT INTO job_files (job_id, file_name, file_content) VALUES ({ph}, {ph}, {ph})"
                            cursor.execute(sql_insert_file, (job_id, key, content))
                            dataframes[key] = pd.read_csv(io.StringIO(content))

                # Handle form parameters
                for key, value in request.form.items():
                    try:
                        params[key] = int(value)
                    except (ValueError, TypeError):
                        try:
                            params[key] = float(value)
                        except (ValueError, TypeError):
                            params[key] = value

                # Insert initial job record
                sql_insert_job = f"INSERT INTO jobs (job_id, status, created_at, parameters) VALUES ({ph}, {ph}, {ph}, {ph})"
                cursor.execute(sql_insert_job, (job_id, 'pending', datetime.utcnow().isoformat(), json.dumps(params)))

    except Exception as e:
        logger.error(f"Error creating job {job_id} in database: {e}", exc_info=True)
        return jsonify({"error": f"Failed to create job record: {e}"}), 500

    # Start job in a background thread
    thread = threading.Thread(target=run_job, args=(job_id, params, dataframes))
    thread.start()

    return jsonify({"job_id": job_id, "status": "pending"}), 201


### MODIFIED: get_job route ###
@app.route('/jobs/<job_id>', methods=['GET'])
def get_job(job_id):
    ph = get_placeholder()
    sql = f"SELECT * FROM jobs WHERE job_id = {ph}"

    with get_db_connection() as conn:
        job_df = pd.read_sql(sql, conn, params=(job_id,))

    if job_df.empty: return jsonify({"error": "Job not found"}), 404

    job = job_df.iloc[0].to_dict()
    job['parameters'] = json.loads(job['parameters']) if job['parameters'] else {}
    return jsonify(job)


### MODIFIED: list_jobs route ###
@app.route('/jobs', methods=['GET'])
def list_jobs():
    sql = "SELECT job_id, status, created_at, started_at, completed_at, error, parameters FROM jobs ORDER BY created_at DESC"
    with get_db_connection() as conn:
        jobs_df = pd.read_sql(sql, conn)

    jobs = []
    for _, row in jobs_df.iterrows():
        job = row.to_dict()
        job['parameters'] = json.loads(job['parameters']) if job['parameters'] else {}
        jobs.append(job)

    return jsonify({"jobs": jobs})


### MODIFIED: get_job_results route ###
@app.route('/jobs/<job_id>/results', methods=['GET'])
def get_job_results(job_id):
    result_type = request.args.get('type', 'summary')
    ph = get_placeholder()
    sql = f"SELECT data FROM results WHERE job_id = {ph} AND result_type = {ph}"

    with get_db_connection() as conn:
        results_df = pd.read_sql(sql, conn, params=(job_id, result_type))

    if results_df.empty: return jsonify({"error": f"No '{result_type}' results found for job {job_id}"}), 404

    return jsonify(json.loads(results_df['data'].iloc[0]))


### MODIFIED: get_job_files route ###
@app.route('/jobs/<job_id>/files', methods=['GET'])
def get_job_files(job_id):
    ph = get_placeholder()
    sql = f"SELECT file_name FROM job_files WHERE job_id = {ph}"
    with get_db_connection() as conn:
        files_df = pd.read_sql(sql, conn, params=(job_id,))

    return jsonify({"files": files_df['file_name'].tolist()})


### MODIFIED: get_job_file_content route ###
@app.route('/jobs/<job_id>/files/<file_name>', methods=['GET'])
def get_job_file_content(job_id, file_name):
    ph = get_placeholder()
    sql = f"SELECT file_content FROM job_files WHERE job_id = {ph} AND file_name = {ph}"

    with get_db_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(sql, (job_id, file_name))
            result = cursor.fetchone()

    if not result: return jsonify({"error": f"File '{file_name}' not found for job {job_id}"}), 404
    return Response(result[0], mimetype='text/plain')


### MODIFIED: Startup block ###
if __name__ == '__main__':
    db_type = 'NeonDB (PostgreSQL)' if USE_NEONDB else 'SQLite'
    db_location = "Loaded from DATABASE_URL env var" if USE_NEONDB else DB_PATH

    logger.info("=" * 50)
    logger.info(f"STARTING FLASK ON PORT {PORT}")
    logger.info(f"ENVIRONMENT: {ENV}")
    logger.info(f"DATABASE TYPE: {db_type}")
    logger.info(f"DATABASE LOCATION: {db_location}")
    logger.info("=" * 50)

    if INITIALIZATION_ERROR:
        logger.critical(f"Application cannot start due to initialization error: {INITIALIZATION_ERROR}")
    else:
        app.run(
            host='0.0.0.0', port=PORT,
            debug=(ENV == 'development'), threaded=True
        )