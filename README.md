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

## RunPod Deployment

Your application is deployed on RunPod with a load-balancing endpoint. Here's how to use it:

### Understanding RunPod Setup

RunPod distributes HTTP requests across available workers. Your container must:
1. Run an HTTP server on the port specified in the `PORT` environment variable (default: 8000)
2. Have a `/ping` health check endpoint on the port specified in `PORT_HEALTH` (default: 8000)

### Current Configuration

The Flask app in `flask_app.py` is configured to:
- Listen on port 8000 (via `PORT` env var)
- Provide `/ping` endpoint for health checks
- Handle job submissions and result retrieval

### Using Your RunPod Endpoint

Replace `YOUR_ENDPOINT_URL` with your actual RunPod endpoint URL:

**Health Check:**
```bash
curl https://YOUR_ENDPOINT_URL/ping
```

**Submit a Job (Web Interface):**
```bash
open https://YOUR_ENDPOINT_URL/web
```

**Submit a Job (API):**
```bash
# Upload CSV files and create a job
curl -X POST https://YOUR_ENDPOINT_URL/jobs \
  -F "files=@data_in/POPULATION.csv" \
  -F "files=@data_in/MORTALITE.csv" \
  -F "nb_an_projection=100" \
  -F "nb_scenarios=100"
```

**Check Job Status:**
```bash
curl https://YOUR_ENDPOINT_URL/jobs/<job_id>
```

**Get Results:**
```bash
curl https://YOUR_ENDPOINT_URL/jobs/<job_id>/results?type=summary
```

**Download Result Files:**
```bash
curl https://YOUR_ENDPOINT_URL/jobs/<job_id>/files/<file_name> -o result.csv
```

### Monitoring RunPod Workers

You can monitor your workers through the RunPod dashboard:
1. Go to your RunPod console
2. Navigate to your endpoint
3. View active workers and their status
4. Scale workers up/down as needed

### Environment Variables

When deploying to RunPod, ensure these are set:
- `PORT=8000` - Main HTTP server port
- `PORT_HEALTH=8000` - Health check port
- `ADMIN_PASSWORD` - Admin password for CLI operations (if needed)
- `ENVIRONMENT=production` - Set to production

### Scaling Considerations

- **GPU Memory**: Each worker has 16GB GPU RAM. Adjust `--max-accounts` if needed
- **Worker Count**: Start with 1-2 workers, scale based on demand
- **Timeout**: Set appropriate timeout for long-running jobs (typically 3600+ seconds)

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

### Running from Docker

You can run the CLI from within a Docker container without installing dependencies locally.

#### Build the Docker image

```bash
docker build -t gpu-actuarial-api:latest .
```

#### Run CLI commands in Docker

Run CLI commands using `docker run` with `uv run` to access installed dependencies:

```bash
# Run a projection job
docker run --gpus all gpu-actuarial-api:latest uv run python cli.py run --years 100 --scenarios 100

# Run asynchronously
docker run --gpus all gpu-actuarial-api:latest uv run python cli.py run --years 100 --scenarios 100 --async

# Check job status
docker run --gpus all gpu-actuarial-api:latest uv run python cli.py status job_20250118_101530_123456

# List all jobs
docker run --gpus all gpu-actuarial-api:latest uv run python cli.py list

# View results
docker run --gpus all gpu-actuarial-api:latest uv run python cli.py results job_20250118_101530_123456 --type summary
```

#### Persist data with Docker volumes

To preserve jobs and results between container runs, mount volumes:

```bash
# Run with persistent database and results
docker run --gpus all \
  -v $(pwd)/jobs.db:/app/jobs.db \
  -v $(pwd)/results:/app/results \
  gpu-actuarial-api:latest \
  uv run python cli.py run --years 100 --scenarios 100

# Check status (data persists)
docker run --gpus all \
  -v $(pwd)/jobs.db:/app/jobs.db \
  -v $(pwd)/results:/app/results \
  gpu-actuarial-api:latest \
  uv run python cli.py list
```

#### Docker Compose for persistent CLI usage

Create a `docker-compose.yml` for easier management:

```yaml
version: '3.8'

services:
  cli:
    image: gpu-actuarial-api:latest
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
      - ADMIN_PASSWORD=admin123
    volumes:
      - ./jobs.db:/app/jobs.db
      - ./results:/app/results
      - ./uploads:/app/uploads
    entrypoint: uv run python cli.py
```

Then run commands:

```bash
# Run a job
docker-compose run --rm cli run --years 100 --scenarios 100

# Check status
docker-compose run --rm cli status job_20250118_101530_123456

# List jobs
docker-compose run --rm cli list

# View results
docker-compose run --rm cli results job_20250118_101530_123456 --type summary
```

#### Important Docker considerations

- **GPU Support**: Use `--gpus all` flag to enable GPU access (requires NVIDIA Docker runtime)
- **Volume Mounts**: Mount `jobs.db` and `results/` directories to persist data across container runs
- **Environment Variables**: Set `ADMIN_PASSWORD` via `-e` flag if needed
- **Working Directory**: CLI runs from `/app` inside the container
- **Data Persistence**: Without volume mounts, job data is lost when the container exits

#### Troubleshooting GPU Access in Docker

If the CLI crashes with CUDA errors but `nvidia-smi` works on your host:

**Step 1: Install NVIDIA Container Toolkit**

```bash
# Add NVIDIA package repositories
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

# Install nvidia-container-toolkit
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# Restart Docker
sudo systemctl restart docker
```

**Step 2: Test GPU access in a test container**

```bash
# This should show your GPU (same as nvidia-smi on host)
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

If this fails, the NVIDIA Container Toolkit isn't properly installed.

**Step 3: Test GPU access in your container**

```bash
# Test CUDA availability
docker run --rm --gpus all gpu-actuarial-api:latest uv run python -c "from numba import cuda; print('CUDA available:', cuda.is_available())"

# Test GPU info
docker run --rm --gpus all gpu-actuarial-api:latest uv run python -c "from numba import cuda; print(cuda.gpus)"
```

**Step 4: Run CLI with GPU**

```bash
# Make sure to use --gpus all flag
docker run --gpus all gpu-actuarial-api:latest uv run python cli.py run --years 10 --scenarios 10 --max-accounts 5
```

**Common Issues:**

- **Missing `--gpus all` flag**: Always include this flag when running the container
- **NVIDIA Container Toolkit not installed**: Install it using the commands above
- **Docker daemon not restarted**: Run `sudo systemctl restart docker` after installing toolkit
- **Insufficient GPU memory**: Use `--max-accounts` to limit memory usage for testing

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
