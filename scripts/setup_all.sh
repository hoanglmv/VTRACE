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
./scripts/setup_3dgs.sh

# 5. Download data
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
