# Production Dockerfile for VGGT-based processing project
# Use the official PyTorch image with CUDA support
# FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime
# FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel

ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# Set working directory
WORKDIR /workspace

ENV TZ=UTC
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get update && apt-get install -y \
    wget \
    curl \
    git \
    build-essential \
    cmake \
    ninja-build \
    ffmpeg \
    ca-certificates \
    ssh \
    && update-ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# install uv
RUN curl -LsSf https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-unknown-linux-gnu.tar.gz | tar -xzC /usr/local/bin --strip-components=1

# Copy project files
COPY . /workspace

RUN uv sync

# Default command
CMD ["uv", "run", "main.py"]
