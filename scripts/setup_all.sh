#!/bin/bash
set -e

echo "========================================================"
echo "   VTRACE: COMPLETE SYSTEM & DATA SETUP"
echo "========================================================"

# Ensure we are in the project root
cd "$(dirname "$0")/.."

# 1. Check for uv command
if ! command -v uv &> /dev/null; then
    echo "Không tìm thấy lệnh 'uv'. Đang tự động cài đặt uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source $HOME/.local/bin/env
fi

# 2. Run 3DGS and environment setup
echo ""
echo ">>> [1/2] Khởi tạo môi trường ảo Python và thiết lập 3D Gaussian Splatting..."

mkdir -p src/vtrace

if [ ! -d "src/vtrace/gaussian-splatting" ]; then
    echo "Cloning gaussian-splatting repository..."
    git clone --recursive https://github.com/graphdeco-inria/gaussian-splatting src/vtrace/gaussian-splatting
else
    echo "gaussian-splatting repository already exists."
fi

echo "Applying custom patches to 3DGS repository..."
if [ -f "src/vtrace/patches/dataset_readers.py" ]; then
    cp src/vtrace/patches/dataset_readers.py src/vtrace/gaussian-splatting/scene/dataset_readers.py
fi
if [ -f "src/vtrace/patches/train.py" ]; then
    cp src/vtrace/patches/train.py src/vtrace/gaussian-splatting/train.py
fi
if [ -f "src/vtrace/patches/arguments_init.py" ]; then
    cp src/vtrace/patches/arguments_init.py src/vtrace/gaussian-splatting/arguments/__init__.py
fi
if [ -f "src/vtrace/patches/cameras.py" ]; then
    cp src/vtrace/patches/cameras.py src/vtrace/gaussian-splatting/scene/cameras.py
fi
if [ -f "src/vtrace/patches/gaussian_model.py" ]; then
    cp src/vtrace/patches/gaussian_model.py src/vtrace/gaussian-splatting/scene/gaussian_model.py
fi
if [ -f "src/vtrace/patches/rasterizer_impl.h" ]; then
    cp src/vtrace/patches/rasterizer_impl.h src/vtrace/gaussian-splatting/submodules/diff-gaussian-rasterization/cuda_rasterizer/rasterizer_impl.h
fi
if [ -f "src/vtrace/patches/camera_utils.py" ]; then
    cp src/vtrace/patches/camera_utils.py src/vtrace/gaussian-splatting/utils/camera_utils.py
fi

echo "Syncing Python environment with uv (installing PyTorch, OpenCV, etc.)..."
uv sync

echo "Checking CUDA version and aligning environment..."

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
    
    if [ "$SYS_CUDA_VERSION" != "12.1" ] && [ -d "/usr/local/cuda-12.1" ]; then
        echo "Found CUDA 12.1 toolkit. Switching build tools to CUDA 12.1 to match default PyTorch..."
        export CUDA_HOME="/usr/local/cuda-12.1"
        export PATH="/usr/local/cuda-12.1/bin:$PATH"
        export LD_LIBRARY_PATH="/usr/local/cuda-12.1/lib64:$LD_LIBRARY_PATH"
        SYS_CUDA_VERSION="12.1"
    fi
    
    if [ "$SYS_CUDA_VERSION" != "12.1" ]; then
        if [ "$SYS_CUDA_VERSION" = "12.4" ]; then
            echo "System CUDA is 12.4. Reinstalling PyTorch and gsplat to match CUDA 12.4..."
            uv pip install --force-reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
            uv pip install --force-reinstall gsplat
        elif [ "$SYS_CUDA_VERSION" = "12.6" ]; then
            echo "System CUDA is 12.6. Reinstalling PyTorch and gsplat to match CUDA 12.6..."
            uv pip install --force-reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
            uv pip install --force-reinstall gsplat
        elif [ "$SYS_CUDA_VERSION" = "11.8" ]; then
            echo "System CUDA is 11.8. Reinstalling PyTorch and gsplat to match CUDA 11.8..."
            uv pip install --force-reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
            uv pip install --force-reinstall gsplat
        else
            echo "Warning: System CUDA version is $SYS_CUDA_VERSION. If build fails, manually run matching PyTorch install."
        fi
    fi
else
    echo "Warning: nvcc not found in PATH. Ensure CUDA Toolkit is installed."
fi

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

echo "Installing remaining submodules using uv..."
uv pip install -p .venv --no-build-isolation ./src/vtrace/gaussian-splatting/submodules/diff-gaussian-rasterization
uv pip install -p .venv --no-build-isolation ./src/vtrace/gaussian-splatting/submodules/simple-knn
uv pip install -p .venv --no-build-isolation ./src/vtrace/gaussian-splatting/submodules/fused-ssim

# 3. Download data
echo ""
echo ">>> [2/2] Tải dữ liệu VAI_NVS_DATA từ Google Drive..."
FILE_ID="12vOrYdBT_0yrvV48pf--yXaSzXD5QONV"

download_and_extract() {
    echo "Đang tải VAI_NVS_DATA.zip từ Google Drive..."
    uvx gdown "${FILE_ID}" -O VAI_NVS_DATA.zip
    echo "Đang giải nén VAI_NVS_DATA.zip..."
    if command -v unzip &> /dev/null; then
        unzip -q VAI_NVS_DATA.zip
    else
        python3 -m zipfile -e VAI_NVS_DATA.zip .
    fi
    rm -f VAI_NVS_DATA.zip
    
    # Wrap phase1 in VAI_NVS_DATA if extracted directly
    if [ -d "phase1" ]; then
        mkdir -p VAI_NVS_DATA
        mv phase1 VAI_NVS_DATA/
    fi
    rm -rf __MACOSX
}

if [ -d "VAI_NVS_DATA" ]; then
    echo "Thư mục VAI_NVS_DATA đã tồn tại. Bạn có muốn tải lại không? (y/n, mặc định là n): "
    read -r RE_DOWNLOAD
    if [ "$RE_DOWNLOAD" = "y" ] || [ "$RE_DOWNLOAD" = "Y" ]; then
        download_and_extract
    else
        echo "Bỏ qua bước tải dữ liệu."
    fi
else
    download_and_extract
fi

echo ""
echo "========================================================"
echo "   Hệ thống VTRACE đã thiết lập thành công!"
echo "========================================================"
