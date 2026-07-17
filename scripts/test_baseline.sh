#!/bin/bash
set -e

echo "======================================"
echo "   VTRACE: TEST BASELINE PIPELINE     "
echo "======================================"

# Ensure we are in the project root
cd "$(dirname "$0")/.."

# Check if command line argument is provided
CHOICE=$1

if [ -z "$CHOICE" ]; then
    echo "Chọn tập dữ liệu muốn chạy test:"
    echo "1) Public Set (VAI_NVS_DATA_ROUND2/phase1/public_set)"
    echo "2) Private Set (VAI_NVS_DATA_ROUND2/phase1/private_set1)"
    read -p "Nhập lựa chọn của bạn (1 hoặc 2, mặc định là 1): " INPUT_CHOICE
    if [ "$INPUT_CHOICE" = "2" ]; then
        CHOICE="private"
    else
        CHOICE="public"
    fi
fi

if [ "$CHOICE" = "private" ] || [ "$CHOICE" = "--private" ] || [ "$CHOICE" = "-p" ]; then
    CONFIG_FILE="config/private_fast.yaml"
else
    CONFIG_FILE="config/public_fast.yaml"
fi

# Load output directory from YAML config for printing
OUTPUT_DIR=$(grep 'output_dir:' "$CONFIG_FILE" | awk '{print $2}' | tr -d '"' | tr -d "'")

echo "Sử dụng cấu hình từ: $CONFIG_FILE"
echo "Bắt đầu chạy thử nghiệm (Test Baseline)..."

uv run python pipeline/run_pipeline.py --config "$CONFIG_FILE"

echo "======================================"
echo "Thử nghiệm hoàn tất! Kết quả được lưu tại: $OUTPUT_DIR"
echo "======================================"
