#!/bin/bash

# Exit on error
set -e

echo "Setting up conda environment for brain registration project..."

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "conda could not be found. Please install Anaconda or Miniconda first."
    exit 1
fi

# Initialize conda for shell interaction
source "$(conda info --base)/etc/profile.d/conda.sh"

# Check CUDA availability
if ! command -v nvidia-smi &> /dev/null; then
    echo "NVIDIA GPU driver not found. Please install NVIDIA drivers first."
    exit 1
fi

# Get CUDA version from nvidia-smi version string
CUDA_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | grep -o '^[0-9]*')
echo "Detected CUDA version: $CUDA_VERSION"

# Use CUDA 11 for older drivers, CUDA 12 for newer ones
if [ "$CUDA_VERSION" -ge 525 ]; then
    echo "Using CUDA 12.x compatible packages"
    sed -i 's/cupy-cuda[0-9]\+x/cupy-cuda12x/g' environment.yml
    sed -i 's/cudatoolkit=[0-9]\+\.[0-9]\+/cudatoolkit=12.1/g' environment.yml
else
    echo "Using CUDA 11.x compatible packages"
    sed -i 's/cupy-cuda[0-9]\+x/cupy-cuda11x/g' environment.yml
    sed -i 's/cudatoolkit=[0-9]\+\.[0-9]\+/cudatoolkit=11.8/g' environment.yml
fi

# Remove existing environment if it exists
conda deactivate 2>/dev/null || true
conda env remove -n brain_reg -y 2>/dev/null || true

# Create conda environment
echo "Creating conda environment..."
conda env create -f environment.yml

# Activate environment
echo "Activating environment..."
conda activate brain_reg

echo "Environment setup complete!"
echo "To activate the environment in a new shell, run: conda activate brain_reg" 