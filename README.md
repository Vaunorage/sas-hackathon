# GPU Actuarial Projections API

Flask API for GPU-accelerated actuarial projections.

## Quick Start

```bash
# Build Docker image
docker build -t your-username/gpu-actuarial-api:latest .

# Run with GPU support
docker run --gpus all -p 5000:5000 your-username/gpu-actuarial-api:latest

# Access the API
curl http://localhost:5000/ping

# Access web interface
open http://localhost:5000/web
```

## Docker Hub

```bash
# Push to Docker Hub
docker login
docker push your-username/gpu-actuarial-api:latest
```

## API Endpoints

- `GET /` - API information
- `GET /ping` - Health check
- `GET /ready` - Readiness probe
- `GET /web` - Web interface
- `POST /jobs` - Create job (upload CSV files)
- `GET /jobs` - List all jobs
- `GET /jobs/<job_id>` - Get job details
- `GET /jobs/<job_id>/results?type=summary|detailed|internal` - Get results
- `GET /jobs/<job_id>/files` - List files
- `GET /jobs/<job_id>/files/<file_name>` - Download file

## Command-Line Interface (CLI)

The `cli.py` script provides a terminal-based interface for managing GPU-accelerated actuarial projection jobs. It allows you to create, monitor, and retrieve results from projection jobs without using the web interface.

### Installation

Ensure all dependencies are installed:
```bash
pip install -r requirements.txt
```

### Basic Usage

```bash
python cli.py <command> [options]
```

Display help:
```bash
python cli.py --help
python cli.py <command> --help
```

### Available Commands

#### `run` - Execute a new projection job

Run a projection job with specified parameters:

```bash
# Run synchronously (waits for completion)
python cli.py run --years 100 --scenarios 100

# Run asynchronously (returns immediately with job ID)
python cli.py run --years 100 --scenarios 100 --async

# Run with limited accounts for testing
python cli.py run --years 50 --scenarios 50 --max-accounts 10

# Run with debug output for a specific account
python cli.py run --years 100 --scenarios 100 --debug-account 12345
```

**Options:**
- `--years` (int, default: 100) - Number of years to project
- `--scenarios` (int, default: 100) - Number of Monte Carlo scenarios
- `--max-accounts` (int, optional) - Limit number of accounts to process
- `--debug-account` (int, optional) - Account ID to generate debug trace for
- `--async` - Run asynchronously (return job ID immediately)

#### `status` - Check job status

Display detailed status of a specific job:

```bash
python cli.py status job_20250118_101530_123456
```

Shows:
- Current status (pending, running, completed, failed)
- Creation and completion timestamps
- Progress (batches processed, percentage)
- Job parameters
- Result files generated
- Error messages (if any)

#### `watch` - Monitor job in real-time

Watch a running job with live progress updates:

```bash
# Monitor with default 2-second update interval
python cli.py watch job_20250118_101530_123456

# Monitor with custom update interval
python cli.py watch job_20250118_101530_123456 --interval 5.0
```

Press `Ctrl+C` to stop watching. Displays:
- Status changes
- Progress updates (current batch / total batches, percentage)
- Completion summary with result files

#### `list` - List all jobs

Display all jobs in a formatted table:

```bash
# List all jobs
python cli.py list

# List only running jobs
python cli.py list --status running

# List only completed jobs
python cli.py list --status completed

# Show only first 20 jobs
python cli.py list --limit 20
```

**Status values:** `pending`, `running`, `completed`, `failed`

#### `results` - View job results

Retrieve and display results from a completed job:

```bash
# View summary results (total present values)
python cli.py results job_20250118_101530_123456 --type summary

# View detailed results by account
python cli.py results job_20250118_101530_123456 --type detailed

# View internal projected cash flows
python cli.py results job_20250118_101530_123456 --type internal

# View results in different formats
python cli.py results job_20250118_101530_123456 --format table
python cli.py results job_20250118_101530_123456 --format csv
python cli.py results job_20250118_101530_123456 --format json

# Limit rows displayed
python cli.py results job_20250118_101530_123456 --limit 50

# Save results to file
python cli.py results job_20250118_101530_123456 --save results.csv --format csv
```

**Result Types:**
- `summary` - VP_FLUX_TOTAL: Total present value across all accounts
- `detailed` - VP_FLUX_COMPTE: Present values by account
- `internal` - FLUX_PROJETES: Projected cash flows by time period

**Filters:**
- `--an-eval` (int) - Filter by year (for internal type)
- `--mois-eval` (int) - Filter by month (for internal type)
- `--id-compte` (int) - Filter by account ID (for detailed type)

**Output Formats:**
- `table` - Formatted table (default)
- `csv` - Semicolon-separated values
- `json` - JSON format

#### `get-all-results` - Retrieve all result types

Retrieve and display all three result types from a completed job:

```bash
# Show all results with default 10 rows per table
python cli.py get-all-results job_20250118_101530_123456

# Show all results with 50 rows per table
python cli.py get-all-results job_20250118_101530_123456 --limit 50
```

Displays:
- FLUX_PROJETES (projected cash flows)
- VP_FLUX_COMPTE (present values by account)
- VP_FLUX_TOTAL (total present value)

#### `clear` - Clear database

Delete all jobs and optionally associated files:

```bash
# Delete all jobs from database (requires confirmation)
python cli.py clear --confirm

# Delete all jobs and associated files
python cli.py clear --confirm --delete-files

# Provide password directly (otherwise prompted)
python cli.py clear --confirm --password admin123 --delete-files
```

**Warning:** This operation is irreversible. Requires admin password (default: `admin123`, configurable via `ADMIN_PASSWORD` environment variable).

#### `info` - Show system information

Display system configuration and job statistics:

```bash
python cli.py info
```

Shows:
- CLI version
- GPU availability status
- Database location
- Data folder paths
- Job counts by status

### Workflow Examples

**Example 1: Run a job and monitor progress**

```bash
# Start job asynchronously
JOB_ID=$(python cli.py run --years 100 --scenarios 100 --async | grep "job_" | awk '{print $NF}')

# Monitor in real-time
python cli.py watch $JOB_ID

# Check final status
python cli.py status $JOB_ID

# View results
python cli.py results $JOB_ID --type summary
```

**Example 2: Run synchronously and save results**

```bash
# Run and wait for completion
python cli.py run --years 50 --scenarios 50

# Get job ID from list
JOB_ID=$(python cli.py list --status completed --limit 1 | grep "job_" | head -1 | awk '{print $1}')

# Save all results
python cli.py results $JOB_ID --type summary --save summary.csv --format csv
python cli.py results $JOB_ID --type detailed --save detailed.csv --format csv
python cli.py results $JOB_ID --type internal --save internal.csv --format csv
```

**Example 3: Debug a specific account**

```bash
# Run with debug output for account 12345
python cli.py run --years 100 --scenarios 100 --debug-account 12345 --async

# Get job ID
JOB_ID=$(python cli.py list --status completed --limit 1 | grep "job_" | head -1 | awk '{print $1}')

# Check results
python cli.py status $JOB_ID
```

### Database

The CLI uses SQLite database (`jobs.db`) to store:
- Job metadata (ID, status, timestamps)
- Job parameters
- Progress information
- Result data tables:
  - `flux_projetes` - Projected cash flows
  - `vp_flux_compte` - Present values by account
  - `vp_flux_total` - Total present values

Results are stored both in the database and as CSV files in the `results/` directory.

### Environment Variables

- `ADMIN_PASSWORD` - Admin password for `clear` command (default: `admin123`)

### Exit Codes

- `0` - Success
- `1` - Error or failure
