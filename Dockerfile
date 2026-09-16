# TwinGuard Reproducible Execution Environment
# Intel Physical AI Online Challenge — Bimanual VLA Manipulation
FROM python:3.13-slim

# Install system dependencies for headless MuJoCo and OpenVINO rendering
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libosmesa6 \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependency specifications first for layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy repository source tree
COPY . .

# Set default environment variables for headless execution
ENV PYTHONUNBUFFERED=1
ENV MUJOCO_GL=osmesa

# Default command: run the complete test suite
CMD ["pytest", "tests/", "-v"]
