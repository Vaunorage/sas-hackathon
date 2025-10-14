import os
import logging
import json
import threading
from datetime import datetime
from flask import Flask, request, jsonify, Response
from flask_cors import CORS  
from werkzeug.exceptions import HTTPException
import pandas as pd
import sqlite3
from paths import HERE
import io

# Import your ACFC algorithm
from test.gpu1 import gpu_acfc_algorithm_complete

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Enable CORS for all routes
CORS(app, resources={
    r"/*": {
        "origins": "*",
        "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"]
    }
})

# Configuration
PORT = int(os.environ.get('PORT', 80))
ENV = os.environ.get('ENVIRONMENT', 'production')
DB_PATH = HERE.joinpath('app/jobs.db')
DATA_PATH = HERE.joinpath("data_in")


# Database initialization
def init_db():
    """Initialize database tables if they don't exist"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Create jobs table if not exists
    cursor.execute('''
                   CREATE TABLE IF NOT EXISTS jobs
                   (
                       job_id
                       TEXT
                       PRIMARY
                       KEY,
                       status
                       TEXT,
                       created_at
                       TEXT,
                       started_at
                       TEXT,
                       completed_at
                       TEXT,
                       error
                       TEXT,
                       parameters
                       TEXT
                   )
                   ''')

    # Create results table if not exists
    cursor.execute('''
                   CREATE TABLE IF NOT EXISTS results
                   (
                       id
                       INTEGER
                       PRIMARY
                       KEY
                       AUTOINCREMENT,
                       job_id
                       TEXT,
                       result_type
                       TEXT,
                       data
                       TEXT,
                       FOREIGN
                       KEY
                   (
                       job_id
                   ) REFERENCES jobs
                   (
                       job_id
                   )
                       )
                   ''')

    # Create job_files table to store uploaded files
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS job_files
        (
            file_id INTEGER PRIMARY KEY AUTOINCREMENT,
            job_id TEXT,
            file_name TEXT,
            file_content TEXT,
            FOREIGN KEY (job_id) REFERENCES jobs (job_id)
        )
    ''')

    conn.commit()
    conn.close()


init_db()


# Job execution function
def run_job(job_id, params, dataframes):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    try:
        # Update status to running
        cursor.execute('''
                       UPDATE jobs
                       SET status     = ?,
                           started_at = ?
                       WHERE job_id = ?
                       ''', ('running', datetime.utcnow().isoformat(), job_id))
        conn.commit()

        # Run the algorithm, passing dataframes directly
        results, detailed_results, internal_results = gpu_acfc_algorithm_complete(
            data_path=DATA_PATH,  # Always provide default path for fallback
            nb_accounts=params.get('nb_accounts', 4),
            nb_scenarios=params.get('nb_scenarios', 10),
            nb_years=params.get('nb_years', 10),
            nb_sc_int=params.get('nb_sc_int', 10),
            nb_an_projection_int=params.get('nb_an_projection_int', 10),
            choc_capital=params.get('choc_capital', 0.35),
            hurdle_rt=params.get('hurdle_rt', 0.10),
            log_account_id=params.get('log_account_id'),
            log_scenario=params.get('log_scenario'),
            log_max_years=params.get('log_max_years'),
            log_internal_scenario=params.get('log_internal_scenario'),
            verbose=False,
            **dataframes
        )

        # Store results
        cursor.execute('''
                       INSERT INTO results (job_id, result_type, data)
                       VALUES (?, ?, ?)
                       ''', (job_id, 'summary', results.to_json(orient='records')))

        cursor.execute('''
                       INSERT INTO results (job_id, result_type, data)
                       VALUES (?, ?, ?)
                       ''', (job_id, 'detailed', detailed_results.to_json(orient='records')))

        cursor.execute('''
                       INSERT INTO results (job_id, result_type, data)
                       VALUES (?, ?, ?)
                       ''', (job_id, 'internal', internal_results.to_json(orient='records')))

        # Update status to completed
        cursor.execute('''
                       UPDATE jobs
                       SET status       = ?,
                           completed_at = ?
                       WHERE job_id = ?
                       ''', ('completed', datetime.utcnow().isoformat(), job_id))

    except Exception as e:
        logger.error(f"Job {job_id} failed: {e}", exc_info=True)
        # Update status to failed
        cursor.execute('''
                       UPDATE jobs
                       SET status       = ?,
                           completed_at = ?,
                           error        = ?
                       WHERE job_id = ?
                       ''', ('failed', datetime.utcnow().isoformat(), str(e), job_id))

    finally:
        conn.commit()
        conn.close()


# Error handlers
@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Resource not found"}), 404


@app.errorhandler(500)
def internal_error(e):
    logger.error(f"Internal server error: {str(e)}")
    return jsonify({"error": "Internal server error"}), 500


@app.errorhandler(Exception)
def handle_exception(e):
    if isinstance(e, HTTPException):
        return jsonify({"error": e.description}), e.code

    logger.error(f"Unhandled exception: {str(e)}", exc_info=True)
    return jsonify({"error": "An unexpected error occurred"}), 500


@app.before_request
def log_request():
    logger.info(f"{request.method} {request.path} - {request.remote_addr}")


@app.after_request
def after_request(response):
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'DENY'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    return response


# Routes
@app.route('/')
def hello():
    return jsonify({
        "message": "ACFC Algorithm API",
        "version": "1.0.0",
        "environment": ENV
    })


@app.route('/ping')
def ping():
    return jsonify({"status": "healthy"}), 200


@app.route('/ready')
def ready():
    return jsonify({"status": "ready"}), 200


@app.route('/jobs', methods=['POST'])
def create_job():
    job_id = f"job_{datetime.utcnow().strftime('%Y%m%d_%H%M%S_%f')}"
    params = {}
    dataframes = {}

    # Handle file uploads by reading them into dataframes and saving to DB
    if request.files:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        for key, file in request.files.items():
            if file and file.filename:
                try:
                    # Read file content once
                    content = file.stream.read().decode('utf-8')
                    
                    # Save to database
                    cursor.execute('''
                        INSERT INTO job_files (job_id, file_name, file_content)
                        VALUES (?, ?, ?)
                    ''', (job_id, key, content))

                    # Use content to create dataframe
                    dataframes[key] = pd.read_csv(io.StringIO(content))

                except Exception as e:
                    conn.close()
                    return jsonify({"error": f"Failed to read or save {file.filename}: {e}"}), 400
        
        conn.commit()
        conn.close()

    # Handle form parameters
    form_params = request.form.to_dict()
    for key, value in form_params.items():
        try:
            params[key] = int(value)
        except (ValueError, TypeError):
            try:
                params[key] = float(value)
            except (ValueError, TypeError):
                params[key] = value

    # We don't store the dataframes in the DB, only the params
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
                   INSERT INTO jobs (job_id, status, created_at, started_at, completed_at, error, parameters)
                   VALUES (?, ?, ?, ?, ?, ?, ?)
                   ''', (job_id, 'pending', datetime.utcnow().isoformat(), None, None, None, json.dumps(params)))
    conn.commit()
    conn.close()

    # Start job in background thread, passing dataframes in memory
    thread = threading.Thread(target=run_job, args=(job_id, params, dataframes))
    thread.start()

    return jsonify({"job_id": job_id, "status": "pending"}), 201


@app.route('/jobs/<job_id>', methods=['GET'])
def get_job(job_id):
    conn = sqlite3.connect(DB_PATH)
    job_df = pd.read_sql('SELECT * FROM jobs WHERE job_id = ?', conn, params=(job_id,))
    conn.close()

    if job_df.empty:
        return jsonify({"error": "Job not found"}), 404

    job = job_df.iloc[0]
    return jsonify({
        "job_id": job['job_id'],
        "status": job['status'],
        "created_at": job['created_at'],
        "started_at": job['started_at'],
        "completed_at": job['completed_at'],
        "error": job['error'],
        "parameters": json.loads(job['parameters'])
    })


@app.route('/jobs', methods=['GET'])
def list_jobs():
    conn = sqlite3.connect(DB_PATH)
    jobs_df = pd.read_sql(
        'SELECT job_id, status, created_at, started_at, completed_at, error, parameters FROM jobs ORDER BY created_at DESC',
        conn
    )
    conn.close()

    # Parse parameters JSON for each job
    jobs = []
    for _, job in jobs_df.iterrows():
        jobs.append({
            'job_id': job['job_id'],
            'status': job['status'],
            'created_at': job['created_at'],
            'started_at': job['started_at'],
            'completed_at': job['completed_at'],
            'error': job['error'],
            'parameters': json.loads(job['parameters']) if job['parameters'] else {}
        })

    return jsonify({"jobs": jobs})


@app.route('/jobs/<job_id>/results', methods=['GET'])
def get_job_results(job_id):
    result_type = request.args.get('type', 'summary')

    conn = sqlite3.connect(DB_PATH)
    results_df = pd.read_sql('''
                           SELECT data FROM results
                           WHERE job_id = ? AND result_type = ?
                           ''', conn, params=(job_id, result_type))
    conn.close()

    if results_df.empty:
        return jsonify({"error": f"No '{result_type}' results found for job {job_id}"}), 404

    # Data is stored as a JSON string, so we need to parse it
    results_data = json.loads(results_df['data'].iloc[0])
    return jsonify(results_data)


@app.route('/jobs/<job_id>/files', methods=['GET'])
def get_job_files(job_id):
    """Get a list of file names associated with a job."""
    conn = sqlite3.connect(DB_PATH)
    files_df = pd.read_sql('SELECT file_name FROM job_files WHERE job_id = ?', conn, params=(job_id,))
    conn.close()

    if files_df.empty:
        return jsonify({"files": []})

    return jsonify({"files": files_df['file_name'].tolist()})


@app.route('/jobs/<job_id>/files/<file_name>', methods=['GET'])
def get_job_file_content(job_id, file_name):
    """Get the content of a specific file for a job."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('SELECT file_content FROM job_files WHERE job_id = ? AND file_name = ?', (job_id, file_name))
    result = cursor.fetchone()
    conn.close()

    if not result:
        return jsonify({"error": f"File '{file_name}' not found for job {job_id}"}), 404

    # Return the content as plain text, which can be rendered as a table
    return Response(result[0], mimetype='text/plain')


if __name__ == '__main__':
    logger.info("=" * 50)
    logger.info(f"STARTING FLASK ON PORT {PORT}")
    logger.info(f"ENVIRONMENT: {ENV}")
    logger.info(f"DATABASE: {DB_PATH}")
    logger.info("=" * 50)

    app.run(
        host='0.0.0.0',
        port=PORT,
        debug=(ENV == 'development'),
        threaded=True
    )