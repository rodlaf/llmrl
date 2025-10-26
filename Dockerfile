# Usage:
# docker build -t ghcr.io/rodlaf/llmrl .
# docker run --gpus all --env-file .env ghcr.io/rodlaf/llmrl

FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    python3.11 \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies first (cached layer)
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy code (changes frequently, separate layer)
COPY . .
ENV PYTHONPATH=/app

ENTRYPOINT ["python3", "examples/cartpole.py"]
