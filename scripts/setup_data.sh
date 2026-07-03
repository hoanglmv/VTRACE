#!/bin/bash
set -e

echo "Setting up data from Google Drive..."

# Ensure we are in the project root
cd "$(dirname "$0")/.."

# The Google Drive folder ID
FOLDER_URL="https://drive.google.com/drive/folders/1TQc6_FNnSnqbwv_EYeusg5zbkf-4lXJF"

echo "Downloading data from Google Drive using gdown..."
# Use uvx to temporarily install and run gdown
# This will download the folder 'VAI_NVS_DATA' directly into the project root
uvx gdown --folder "${FOLDER_URL}"

echo "Data setup complete! Dữ liệu đã được tải vào thư mục VAI_NVS_DATA."
