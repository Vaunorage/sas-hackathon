FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Copy pyproject.toml and install ALL dependencies
COPY pyproject.toml .
RUN uv pip install --system -e .

# Copy the entire app directory
COPY app ./app

# Add debug output
RUN python --version
RUN python -c "import flask; print(f'Flask version: {flask.__version__}')"

CMD ["python", "app/app.py"]