#!/bin/bash
set -e

echo "Setting up 3D Gaussian Splatting..."

# Ensure we are in the project root
cd "$(dirname "$0")/.."

mkdir -p src/vtrace

if [ ! -d "src/vtrace/gaussian-splatting" ]; then
    echo "Cloning gaussian-splatting repository..."
    git clone --recursive https://github.com/graphdeco-inria/gaussian-splatting src/vtrace/gaussian-splatting
else
    echo "gaussian-splatting repository already exists."
fi

# Apply patches to the cloned repository
echo "Applying custom patches to 3DGS repository..."
if [ -f "src/vtrace/patches/dataset_readers.py" ]; then
    cp src/vtrace/patches/dataset_readers.py src/vtrace/gaussian-splatting/scene/dataset_readers.py
fi
if [ -f "src/vtrace/patches/train.py" ]; then
    cp src/vtrace/patches/train.py src/vtrace/gaussian-splatting/train.py
fi
if [ -f "src/vtrace/patches/rasterizer_impl.h" ]; then
    cp src/vtrace/patches/rasterizer_impl.h src/vtrace/gaussian-splatting/submodules/diff-gaussian-rasterization/cuda_rasterizer/rasterizer_impl.h
fi

echo "Syncing Python environment with uv (installing PyTorch, OpenCV, etc.)..."
uv sync

echo "Installing submodules using uv..."
# Install submodules without build isolation so they use the installed PyTorch
uv pip install -p .venv --no-build-isolation ./src/vtrace/gaussian-splatting/submodules/diff-gaussian-rasterization
uv pip install -p .venv --no-build-isolation ./src/vtrace/gaussian-splatting/submodules/simple-knn
uv pip install -p .venv --no-build-isolation ./src/vtrace/gaussian-splatting/submodules/fused-ssim

echo "Setup completed successfully."
