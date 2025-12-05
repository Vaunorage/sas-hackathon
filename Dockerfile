FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Install system dependencies and uv
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && pip install --no-cache-dir uv \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy pyproject.toml, uv.lock, and README for installation
COPY pyproject.toml uv.lock README.md ./

# Use uv sync to install packages from lockfile
RUN uv sync --frozen

RUN uv pip install --extra-index-url=https://pypi.nvidia.com \
    "cudf-cu12==25.10.*" "dask-cudf-cu12==25.10.*" "cuml-cu12==25.10.*" \
    "cugraph-cu12==25.10.*" "nx-cugraph-cu12==25.10.*" "cuxfilter-cu12==25.10.*" \
    "cucim-cu12==25.10.*" "pylibraft-cu12==25.10.*" "raft-dask-cu12==25.10.*" \
    "cuvs-cu12==25.10.*" "nx-cugraph-cu12==25.10.*"

# Reinstall compatible numba/llvmlite versions after RAPIDS (which may override them)
RUN uv pip install --force-reinstall "llvmlite==0.43.0" "numba==0.60.0" "numpy<2.0"

# Copy application files
COPY flask_app.py .
COPY cli.py .
COPY calculations/gpu.py .
COPY cpu.py .
COPY paths.py .
COPY extract_csv_from_zips.py .
COPY runpod_worker.py .

# Create directories for data (runpod_worker uses temp dirs, but these are useful for debugging)
RUN mkdir -p uploads results static data_in data_out

# Copy zip files from default_data
COPY default_data/*.zip ./default_data/

# Extract CSV files from zips during build
RUN uv run python extract_csv_from_zips.py

# Remove zip files to reduce image size
RUN rm -rf default_data/*.zip

# Copy web interface
COPY static/index.html ./static/

# Expose port (default 80, can be overridden)
EXPOSE 80

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV ENVIRONMENT=production
ENV PORT=80
ENV PORT_HEALTH=80
ENV RUNPOD_CORS=true

# Run the RunPod serverless worker
CMD uv run python runpod_worker.py
