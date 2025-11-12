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
