#!/bin/bash
set -e

echo "======================================"
echo "   VTRACE: TEST BASELINE PIPELINE     "
echo "======================================"

# Ensure we are in the project root
cd "$(dirname "$0")/.."

DATA_DIR="./VAI_NVS_DATA/phase1/public_set"
OUTPUT_DIR="./output_test"

if [ ! -d "$DATA_DIR" ]; then
    echo "Lỗi: Không tìm thấy thư mục dữ liệu $DATA_DIR"
    echo "Hãy chắc chắn rằng bạn đã tải dữ liệu bằng ./scripts/setup_data.sh"
    exit 1
fi

echo "Bắt đầu chạy thử nghiệm (Test Baseline)..."
echo "Số iterations: 100 (để test nhanh)"

uv run python pipeline/run_pipeline.py \
    --data-dir "$DATA_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --iterations 100 \
    --resolution 2

echo "======================================"
echo "Thử nghiệm hoàn tất! Kết quả được lưu tại: $OUTPUT_DIR"
echo "======================================"
