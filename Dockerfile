FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Copy pyproject.toml and install dependencies
COPY pyproject.toml .
RUN uv pip install --system flask

# Copy the entire app directory
COPY app ./app

# Run the app from the app directory
CMD ["python", "app/app.py"]