#!/bin/bash
set -e

echo "======================================"
echo "   VTRACE: FULL BASELINE PIPELINE     "
echo "======================================"

# Ensure we are in the project root
cd "$(dirname "$0")/.."

DATA_DIR="./VAI_NVS_DATA/phase1/public_set"
OUTPUT_DIR="./output"

if [ ! -d "$DATA_DIR" ]; then
    echo "Lỗi: Không tìm thấy thư mục dữ liệu $DATA_DIR"
    echo "Hãy chắc chắn rằng bạn đã tải dữ liệu bằng ./scripts/setup_data.sh"
    exit 1
fi

echo "Bắt đầu chạy huấn luyện đầy đủ (Full Baseline)..."
echo "Số iterations: 30000"

uv run python pipeline/run_pipeline.py \
    --data-dir "$DATA_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --iterations 30000 \
    --resolution 1

echo "======================================"
echo "Huấn luyện hoàn tất! Kết quả được nén tại: $OUTPUT_DIR/submission.zip"
echo "======================================"
