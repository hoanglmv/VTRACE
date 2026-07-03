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

echo "Installing submodules using uv..."
# Install plyfile and submodules
uv pip install plyfile
uv pip install ./src/vtrace/gaussian-splatting/submodules/diff-gaussian-rasterization
uv pip install ./src/vtrace/gaussian-splatting/submodules/simple-knn

echo "Setup completed successfully."
