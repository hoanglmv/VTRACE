#!/bin/bash
set -e

echo "========================================================"
echo "   VTRACE: COMPLETE SYSTEM & DATA SETUP"
echo "========================================================"

# Ensure we are in the project root
cd "$(dirname "$0")/.."

# 1. Check for uv command
if ! command -v uv &> /dev/null; then
    echo "Lỗi: Không tìm thấy lệnh 'uv'."
    echo "Vui lòng cài đặt uv trước bằng lệnh: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

# 2. Syncing Python environment
echo ""
echo ">>> [1/4] Khởi tạo môi trường ảo Python bằng uv..."
uv sync

# 3. Setup 3D Gaussian Splatting submodules
echo ""
echo ">>> [2/4] Thiết lập mô hình 3D Gaussian Splatting..."
mkdir -p src/vtrace

if [ ! -d "src/vtrace/gaussian-splatting" ]; then
    echo "Đang clone kho lưu trữ gaussian-splatting từ graphdeco-inria..."
    git clone --recursive https://github.com/graphdeco-inria/gaussian-splatting src/vtrace/gaussian-splatting
else
    echo "Kho lưu trữ gaussian-splatting đã tồn tại."
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

# 4. Install CUDA submodules with uv
echo ""
echo ">>> [3/4] Biên dịch và cài đặt các CUDA submodules (diff-gaussian-rasterization, simple-knn, fused-ssim)..."
uv pip install -p .venv --no-build-isolation ./src/vtrace/gaussian-splatting/submodules/diff-gaussian-rasterization
uv pip install -p .venv --no-build-isolation ./src/vtrace/gaussian-splatting/submodules/simple-knn
uv pip install -p .venv --no-build-isolation ./src/vtrace/gaussian-splatting/submodules/fused-ssim

# 5. Download data
echo ""
echo ">>> [4/4] Tải dữ liệu VAI_NVS_DATA từ Google Drive..."
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
