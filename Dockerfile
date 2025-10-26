# Usage:
# docker build -t llamagym .
# docker run --gpus all -e HF_TOKEN=$HF_TOKEN -e WANDB_API_KEY=$WANDB_API_KEY \
#   llamagym --episodes 500 --batch-size 4 --model google/gemma-2-270m --dtype float16

FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    python3.11 \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY . .
RUN pip3 install --no-cache-dir -e .

ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

ENTRYPOINT ["python3", "examples/cartpole.py"]
