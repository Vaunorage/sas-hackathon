FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Copy requirements and install with uv
COPY requirements.txt .
RUN uv pip install --system -r requirements.txt

COPY app.py .

CMD ["python", "app.py"]
