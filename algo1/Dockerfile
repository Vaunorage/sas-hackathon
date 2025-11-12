FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

COPY ../pyproject.toml .
RUN uv pip install --system -e .

# Copy ALL necessary files
COPY ../paths.py .
COPY app ./app
COPY test ./test
COPY ../data_in ./data_in

# Add /app to PYTHONPATH so paths.py can be imported
ENV PYTHONPATH=/app:$PYTHONPATH

# Simple startup
CMD ["gunicorn", "--bind", "0.0.0.0:80", "--workers", "4", "--threads", "2", "--timeout", "300", "--access-logfile", "-", "--error-logfile", "-", "app.app:app"]
