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

# Copy application files
COPY flask_app.py .
COPY cli.py .
COPY gpu.py .
COPY cpu.py .
COPY paths.py .
COPY extract_csv_from_zips.py .

# Create directories for data
RUN mkdir -p uploads results static data_in

# Copy zip files from default_data
COPY default_data/*.zip ./default_data/

# Extract CSV files from zips during build
RUN uv run python extract_csv_from_zips.py

# Remove zip files to reduce image size
RUN rm -rf default_data/*.zip

# Copy web interface
COPY static/index.html ./static/

# Expose port
EXPOSE 80

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV ENVIRONMENT=production
ENV PORT=80

# Run the Flask application using gunicorn
CMD ["uv", "run", "gunicorn", "--bind", "0.0.0.0:80", "--workers", "4", "--timeout", "300", "--worker-class", "sync", "flask_app:app"]
