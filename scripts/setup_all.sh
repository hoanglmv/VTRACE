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

# Patch rasterizer_impl.h to add #include <cstdint> if missing (fixes build with newer compilers)
PATCH_FILE="src/vtrace/gaussian-splatting/submodules/diff-gaussian-rasterization/cuda_rasterizer/rasterizer_impl.h"
if [ -f "$PATCH_FILE" ]; then
    if ! grep -q "<cstdint>" "$PATCH_FILE"; then
        echo "Patching rasterizer_impl.h with <cstdint>..."
        # Insert #include <cstdint> after #include <vector>
        sed -i '/#include <vector>/a #include <cstdint>' "$PATCH_FILE"
    fi
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
    uvx gdown --id "${FILE_ID}" -O VAI_NVS_DATA.zip
    echo "Đang giải nén VAI_NVS_DATA.zip..."
    if command -v unzip &> /dev/null; then
        unzip -q VAI_NVS_DATA.zip
    else
        python3 -m zipfile -e VAI_NVS_DATA.zip .
    fi
    rm -f VAI_NVS_DATA.zip
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
