FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

# Set working directory
WORKDIR /app

# Copy pyproject.toml and README for installation
COPY pyproject.toml README.md ./
RUN pip install --no-cache-dir .

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
RUN python extract_csv_from_zips.py

# Optional: Remove zip files to reduce image size
# RUN rm -rf default_data

# Copy web interface
COPY static/index.html ./static/

# Expose port
EXPOSE 5000

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV ENVIRONMENT=production

# Run the Flask application
CMD ["python", "flask_app.py"]
