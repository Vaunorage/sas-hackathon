FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Copy pyproject.toml and install dependencies with uv
COPY pyproject.toml .
RUN uv pip install --system -e .

COPY app.py .

CMD ["python", "app.py"]