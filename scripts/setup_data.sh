#!/bin/bash
set -e

echo "Setting up data from Google Drive..."

# Ensure we are in the project root
cd "$(dirname "$0")/.."

FILE_ID="12vOrYdBT_0yrvV48pf--yXaSzXD5QONV"

echo "Downloading VAI_NVS_DATA.zip from Google Drive using gdown..."
uvx gdown --id "${FILE_ID}" -O VAI_NVS_DATA.zip

echo "Extracting VAI_NVS_DATA.zip..."
if command -v unzip &> /dev/null; then
    unzip -q VAI_NVS_DATA.zip
else
    python3 -m zipfile -e VAI_NVS_DATA.zip .
fi

rm -f VAI_NVS_DATA.zip

echo "Data setup complete! Dữ liệu đã được giải nén vào thư mục VAI_NVS_DATA."
