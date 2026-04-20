# Use NVIDIA CUDA base image
FROM nvidia/cuda:11.8-cudnn8-devel-ubuntu20.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.10 as default
RUN ln -s /usr/bin/python3.10 /usr/bin/python

# Upgrade pip
RUN pip install --upgrade pip

# Set working directory
WORKDIR /workspace

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy the project
COPY . .

# Set environment variables for nnUNet
ENV nnUNet_raw="/workspace/data/nnUNet_raw"
ENV nnUNet_preprocessed="/workspace/data/nnUNet_preprocessed"
ENV nnUNet_results="/workspace/nnUNet_results"

# Create necessary directories
RUN mkdir -p $nnUNet_raw $nnUNet_preprocessed $nnUNet_results

# Default command
CMD ["/bin/bash"]