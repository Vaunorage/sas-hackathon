FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Copy pyproject.toml and install ALL dependencies
COPY pyproject.toml .
RUN uv pip install --system -e .

# Copy the entire app directory
COPY app ./app

# Debug: List what files we have
RUN echo "=== Files in /app ===" && ls -la
RUN echo "=== Files in /app/app ===" && ls -la app/

# Test if we can run Python
RUN python --version

# Add a startup script for debugging
RUN echo '#!/bin/bash\n\
echo "========================================"\n\
echo "CONTAINER STARTING"\n\
echo "========================================"\n\
echo "Python version:"\n\
python --version\n\
echo "Files in /app:"\n\
ls -la /app\n\
echo "Files in /app/app:"\n\
ls -la /app/app\n\
echo "========================================"\n\
echo "STARTING FLASK"\n\
echo "========================================"\n\
exec python app/app.py' > /start.sh && chmod +x /start.sh

CMD ["/start.sh"]