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

# --- CUDA Version Auto-Detection & PyTorch Alignment ---
echo "Checking CUDA version and aligning environment..."

# Make sure CUDA_HOME is set if /usr/local/cuda exists
if [ -z "$CUDA_HOME" ]; then
    if [ -d "/usr/local/cuda" ]; then
        export CUDA_HOME="/usr/local/cuda"
    elif [ -d "/usr/local/cuda-12.1" ]; then
        export CUDA_HOME="/usr/local/cuda-12.1"
    fi
fi

if command -v nvcc &> /dev/null; then
    SYS_CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $5}' | cut -d, -f1)
    echo "Detected system NVCC version: $SYS_CUDA_VERSION"
    
    # If the system NVCC is not 12.1, but /usr/local/cuda-12.1 is available,
    # we can use the 12.1 compiler to match PyTorch's default build.
    if [ "$SYS_CUDA_VERSION" != "12.1" ] && [ -d "/usr/local/cuda-12.1" ]; then
        echo "Found CUDA 12.1 toolkit. Switching build tools to CUDA 12.1 to match default PyTorch..."
        export CUDA_HOME="/usr/local/cuda-12.1"
        export PATH="/usr/local/cuda-12.1/bin:$PATH"
        export LD_LIBRARY_PATH="/usr/local/cuda-12.1/lib64:$LD_LIBRARY_PATH"
        SYS_CUDA_VERSION="12.1"
    fi
    
    # If we are stuck with a different system CUDA version (no 12.1 toolkit on disk),
    # we reinstall PyTorch to match the system CUDA toolkit version.
    if [ "$SYS_CUDA_VERSION" != "12.1" ]; then
        if [ "$SYS_CUDA_VERSION" = "12.4" ]; then
            echo "System CUDA is 12.4. Reinstalling PyTorch to match CUDA 12.4..."
            uv pip install --force-reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
        elif [ "$SYS_CUDA_VERSION" = "12.6" ]; then
            echo "System CUDA is 12.6. Reinstalling PyTorch to match CUDA 12.6..."
            uv pip install --force-reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
        elif [ "$SYS_CUDA_VERSION" = "11.8" ]; then
            echo "System CUDA is 11.8. Reinstalling PyTorch to match CUDA 11.8..."
            uv pip install --force-reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
        else
            echo "Warning: System CUDA version is $SYS_CUDA_VERSION. If build fails, manually run matching PyTorch install."
        fi
    fi
else
    echo "Warning: nvcc not found in PATH. Ensure CUDA Toolkit is installed."
fi

# --- CUDA version check bypass ---
echo "Patching PyTorch's cpp_extension.py to bypass CUDA mismatch verification..."
.venv/bin/python -c '
import torch.utils.cpp_extension as m
path = m.__file__
with open(path, "r") as f:
    code = f.read()
target = "raise RuntimeError(CUDA_MISMATCH_MESSAGE"
if target in code:
    print("Found CUDA mismatch check, patching it...")
    code = code.replace(target, "print(\"Warning: CUDA mismatch warning bypassed\"); # raise RuntimeError(CUDA_MISMATCH_MESSAGE")
    with open(path, "w") as f:
        f.write(code)
    print("Patch applied successfully.")
else:
    print("CUDA mismatch check not found or already patched.")
'

echo "Installing submodules using uv..."
# Install submodules without build isolation so they use the installed PyTorch
uv pip install -p .venv --no-build-isolation ./src/vtrace/gaussian-splatting/submodules/diff-gaussian-rasterization
uv pip install -p .venv --no-build-isolation ./src/vtrace/gaussian-splatting/submodules/simple-knn
uv pip install -p .venv --no-build-isolation ./src/vtrace/gaussian-splatting/submodules/fused-ssim

echo "Setup completed successfully."
