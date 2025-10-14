import os
import logging
import json
import threading
from datetime import datetime
from flask import Flask, jsonify, request
from werkzeug.exceptions import HTTPException
import pandas as pd
import sqlite3
from paths import HERE

# Import your ACFC algorithm
from test.gpu1 import gpu_acfc_algorithm_complete

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Configuration
PORT = int(os.environ.get('PORT', 80))
ENV = os.environ.get('ENVIRONMENT', 'production')
DB_PATH = HERE.joinpath('app/jobs.db')


# Database initialization
def init_db():
    conn = sqlite3.connect(DB_PATH)

    # Create jobs table using pandas
    jobs_df = pd.DataFrame(columns=[
        'job_id', 'status', 'created_at', 'started_at',
        'completed_at', 'error', 'parameters'
    ])
    jobs_df.to_sql('jobs', conn, if_exists='fail', index=False)

    # Create results table using pandas
    results_df = pd.DataFrame(columns=['job_id', 'result_type', 'data'])
    results_df.to_sql('results', conn, if_exists='fail', index=False)

    conn.close()


init_db()


# Job execution function
def run_job(job_id, params):
    conn = sqlite3.connect(DB_PATH)

    try:
        # Update status to running
        jobs_df = pd.read_sql('SELECT * FROM jobs WHERE job_id = ?', conn, params=(job_id,))
        jobs_df.loc[0, 'status'] = 'running'
        jobs_df.loc[0, 'started_at'] = datetime.utcnow().isoformat()
        jobs_df.to_sql('jobs', conn, if_exists='replace', index=False)

        # Run the algorithm
        results, detailed_results, internal_results = gpu_acfc_algorithm_complete(
            data_path=params.get('data_path', '.'),
            nb_accounts=params.get('nb_accounts', 4),
            nb_scenarios=params.get('nb_scenarios', 10),
            nb_years=params.get('nb_years', 10),
            nb_sc_int=params.get('nb_sc_int', 10),
            nb_an_projection_int=params.get('nb_an_projection_int', 10),
            choc_capital=params.get('choc_capital', 0.35),
            hurdle_rt=params.get('hurdle_rt', 0.10),
            verbose=False
        )

        # Store results
        results_data = pd.DataFrame([
            {'job_id': job_id, 'result_type': 'summary', 'data': results.to_json(orient='records')},
            {'job_id': job_id, 'result_type': 'detailed', 'data': detailed_results.to_json(orient='records')},
            {'job_id': job_id, 'result_type': 'internal', 'data': internal_results.to_json(orient='records')}
        ])
        results_data.to_sql('results', conn, if_exists='append', index=False)

        # Update status to completed
        jobs_df = pd.read_sql('SELECT * FROM jobs WHERE job_id = ?', conn, params=(job_id,))
        jobs_df.loc[0, 'status'] = 'completed'
        jobs_df.loc[0, 'completed_at'] = datetime.utcnow().isoformat()
        jobs_df.to_sql('jobs', conn, if_exists='replace', index=False)

    except Exception as e:
        logger.error(f"Job {job_id} failed: {str(e)}")
        jobs_df = pd.read_sql('SELECT * FROM jobs WHERE job_id = ?', conn, params=(job_id,))
        jobs_df.loc[0, 'status'] = 'failed'
        jobs_df.loc[0, 'error'] = str(e)
        jobs_df.loc[0, 'completed_at'] = datetime.utcnow().isoformat()
        jobs_df.to_sql('jobs', conn, if_exists='replace', index=False)
    finally:
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
    params = request.json or {}
    job_id = f"job_{datetime.utcnow().strftime('%Y%m%d_%H%M%S_%f')}"

    conn = sqlite3.connect(DB_PATH)
    job_df = pd.DataFrame([{
        'job_id': job_id,
        'status': 'pending',
        'created_at': datetime.utcnow().isoformat(),
        'started_at': None,
        'completed_at': None,
        'error': None,
        'parameters': json.dumps(params)
    }])
    job_df.to_sql('jobs', conn, if_exists='append', index=False)
    conn.close()

    # Start job in background thread
    thread = threading.Thread(target=run_job, args=(job_id, params))
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
def get_results(job_id):
    result_type = request.args.get('type', 'summary')

    conn = sqlite3.connect(DB_PATH)

    # Check job exists and is completed
    job_df = pd.read_sql('SELECT status FROM jobs WHERE job_id = ?', conn, params=(job_id,))

    if job_df.empty:
        conn.close()
        return jsonify({"error": "Job not found"}), 404

    if job_df.iloc[0]['status'] != 'completed':
        conn.close()
        return jsonify({"error": f"Job status is {job_df.iloc[0]['status']}, not completed"}), 400

    # Get results
    results_df = pd.read_sql(
        'SELECT data FROM results WHERE job_id = ? AND result_type = ?',
        conn,
        params=(job_id, result_type)
    )
    conn.close()

    if results_df.empty:
        return jsonify({"error": f"No results found for type {result_type}"}), 404

    return jsonify({
        "job_id": job_id,
        "type": result_type,
        "data": json.loads(results_df.iloc[0]['data'])
    })


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