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

# Patch rasterizer_impl.h to add #include <cstdint> if missing (fixes build with newer compilers)
PATCH_FILE="src/vtrace/gaussian-splatting/submodules/diff-gaussian-rasterization/cuda_rasterizer/rasterizer_impl.h"
if [ -f "$PATCH_FILE" ]; then
    if ! grep -q "<cstdint>" "$PATCH_FILE"; then
        echo "Patching rasterizer_impl.h with <cstdint>..."
        # Insert #include <cstdint> after #include <vector>
        sed -i '/#include <vector>/a #include <cstdint>' "$PATCH_FILE"
    fi
fi

echo "Syncing Python environment with uv (installing PyTorch, OpenCV, etc.)..."
uv sync

echo "Installing submodules using uv..."
# Install submodules without build isolation so they use the installed PyTorch
uv pip install -p .venv --no-build-isolation ./src/vtrace/gaussian-splatting/submodules/diff-gaussian-rasterization
uv pip install -p .venv --no-build-isolation ./src/vtrace/gaussian-splatting/submodules/simple-knn
uv pip install -p .venv --no-build-isolation ./src/vtrace/gaussian-splatting/submodules/fused-ssim

echo "Setup completed successfully."
