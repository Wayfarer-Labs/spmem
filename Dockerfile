# Production Dockerfile for VGGT-based processing project
# Use the official PyTorch image with CUDA support
# FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime
FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

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

# Apply VGGT patches for float32 auto-conversion fix
# Patch 1: Fix dpt_head.py - ensure positional embedding maintains dtype
RUN sed -i 's/return x + pos_embed/return (x + pos_embed).to(x.dtype)/' /workspace/.venv/lib/python*/site-packages/vggt/heads/dpt_head.py

# Patch 2: Fix aggregator.py - add dtype detection and conversion for patch tokens
RUN sed -i '/patch_tokens = self.patch_embed(images)/a\\n        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16' /workspace/.venv/lib/python*/site-packages/vggt/models/aggregator.py && \
    sed -i 's/patch_tokens = patch_tokens\["x_norm_patchtokens"\]/patch_tokens = patch_tokens["x_norm_patchtokens"].to(dtype)/' /workspace/.venv/lib/python*/site-packages/vggt/models/aggregator.py && \
    sed -i '/patch_tokens = patch_tokens\["x_norm_patchtokens"\].to(dtype)/a\        else:\n            patch_tokens = patch_tokens.to(dtype)' /workspace/.venv/lib/python*/site-packages/vggt/models/aggregator.py

# Default command
CMD ["uv", "run", "main.py"]
