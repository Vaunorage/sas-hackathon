FROM pytorch/pytorch:2.4.0-cuda12.4-runtime-ubuntu22.04

# Set working directory
WORKDIR /app

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
EXPOSE 5000

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV ENVIRONMENT=production

# Run the Flask application using uv
CMD ["uv", "run", "python", "flask_app.py"]
