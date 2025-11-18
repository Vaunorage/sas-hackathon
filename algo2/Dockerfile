FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

# Set working directory
WORKDIR /app

# Install uv
RUN pip install --no-cache-dir uv

# Copy pyproject.toml, uv.lock, and README for installation
COPY pyproject.toml uv.lock README.md ./

# Use uv sync to install packages from lockfile
RUN uv sync --frozen

# Copy application files
COPY flask_app.py .
COPY gpu.py .
COPY cpu.py .
COPY paths.py .
COPY extract_csv_from_zips.py .

# Create directories for data
RUN mkdir -p uploads results static data_in default_data

# Copy zip files
COPY *.zip ./default_data/

# Extract CSV files from zips during build
RUN uv run python extract_csv_from_zips.py

# Optional: Remove zip files to reduce image size (uncomment to save space)
# RUN rm -rf default_data/*.zip

# Copy web interface
COPY static/index.html ./static/

# Expose port 80 (RunPod default for load balancer)
EXPOSE 80

# Set environment variables for RunPod
ENV PYTHONUNBUFFERED=1
ENV ENVIRONMENT=production
ENV PORT=80

# Health check for RunPod monitoring
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:80/ping || exit 1

# Run the Flask application using uv
CMD ["uv", "run", "python", "flask_app.py"]
