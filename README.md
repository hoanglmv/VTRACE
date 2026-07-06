# VTRACE - Viettel AI Race: Novel View Synthesis (3DGS)

Dự án này là mã nguồn tự động hoá việc huấn luyện (training) và sinh ảnh góc nhìn mới (inference) sử dụng phương pháp **3D Gaussian Splatting (MCMC Optimization với lõi gsplat)** dành riêng cho tập dữ liệu của cuộc thi Viettel AI Race (Digital Twin).

## 🗂️ Cấu trúc dự án (Src-layout)

- `src/vtrace/`: Package chứa mã nguồn cốt lõi.
  - `data_utils.py`: Đọc cấu trúc thư mục COLMAP và xử lý dữ liệu đầu vào.
  - `trainer.py`: Gọi vòng lặp huấn luyện tối ưu hóa bằng MCMC Strategy.
  - `renderer.py`: Sinh ảnh dựa vào file `test_poses.csv` (Quaternion/Translation) và đóng gói submission.
- `scripts/`: Chứa các script hệ thống.
  - `setup_all.sh`: Tự động setup toàn bộ (Cài uv, tải gsplat, bắt lỗi mismatch CUDA và tự động tải dữ liệu giải nén từ Google Drive).
- `notebooks/`: Chứa các Jupyter Notebooks giúp quá trình chạy trở nên trực quan.
  - `01_data_analysis_and_training.ipynb`: Phân tích dữ liệu & Train.
  - `02_inference_and_submission.ipynb`: Render ảnh test & nén thành `submission.zip`.
- `VAI_NVS_DATA/`: Thư mục lưu dữ liệu thi. Script setup sẽ tự động tải cho bạn.

---

## 🚀 Hướng dẫn cài đặt và sử dụng

### A. Quy trình chạy trên Vast.ai (Rút gọn & Tự động hoá)

Vì việc huấn luyện yêu cầu GPU NVIDIA mạnh (khuyên dùng RTX 3090 / 4090), quy trình chạy tối ưu nhất trên **Vast.ai** như sau:

#### 1. Thuê máy chủ GPU
- Lên trang web Vast.ai, chọn thuê một instance **RTX 3090 hoặc RTX 4090** (24GB VRAM).
- Chọn Docker Image có sẵn PyTorch và CUDA (ví dụ: `pytorch/pytorch:2.3.1-cuda12.1-cudnn8-devel` hoặc sử dụng mặc định có Jupyter Lab).

#### 2. Kết nối và Setup tự động
- Mở Terminal của máy chủ vừa thuê (hoặc thông qua Jupyter Lab Terminal).
- Kéo mã nguồn (git pull) và di chuyển vào thư mục VTRACE.
- Chạy script cài đặt tất cả trong một (DUY NHẤT 1 LỆNH CHẠY):
  ```bash
  ./scripts/setup_all.sh
  ```
  *Lệnh này sẽ tự động: Cài đặt trình quản lý `uv`, đồng bộ thư viện lõi `gsplat`, tự động kiểm tra và triệt tiêu lỗi CUDA mismatch (nếu có), đồng thời kéo tập dữ liệu VAI_NVS_DATA từ Google Drive.*

#### 3. Chạy Huấn luyện và Render Pipeline
Sau khi cài đặt xong, hãy chạy toàn bộ pipeline (tự động ước lượng Depth, huấn luyện mô hình MCMC, render ảnh và đóng gói kết quả):
- Chạy trên toàn bộ tập **Private Set**:
  ```bash
  uv run python pipeline/run_pipeline.py --config config/private_high.yaml
  ```
- Chạy riêng biệt cho 1 Scene cụ thể:
  ```bash
  uv run python pipeline/run_pipeline.py --scene VAI_NVS_DATA/phase1/private_set1/HCM0249 --config config/private_high.yaml --mode train
  ```

#### 4. Tải kết quả nộp bài
- Khi pipeline chạy xong, file nén `submission_round1.zip` sẽ nằm trong thư mục `output_private_high/` hoặc `output_public_high/`.
- Chỉ cần click chuột phải vào file này trong cột thư mục bên trái của Jupyter Lab và chọn **Download** để tải về máy cá nhân của bạn.

---

### B. Quy trình chạy thủ công trên Laptop (Không cần GPU - Để phát triển code)

Nếu bạn chỉ chỉnh sửa code trên laptop cá nhân (không có GPU):
1. **Thiết lập code:** Chạy `uv sync` để cài đặt môi trường ảo Python.
2. **Chỉnh sửa:** Thực hiện chỉnh sửa code trực tiếp trên laptop và đẩy lên GitHub (hoặc dùng `scp`/`rsync` để đồng bộ code lên Vast.ai).

---

## 🛠️ Xử lý sự cố thường gặp (Troubleshooting)

- **Lỗi `Found no NVIDIA driver on your system`**: Lỗi này xảy ra khi bạn cố tình chạy huấn luyện/render trực tiếp trên laptop cá nhân không có GPU NVIDIA. Hãy đảm bảo bạn chỉ chạy lệnh chạy huấn luyện trên server Vast.ai.
- **Lỗi vRAM (OOM) khi sinh Depth**: Model depth đã được cấu hình chạy theo cơ chế **Batched Inference (Batch size = 8)** rất tối ưu cho RTX 3090/4090. Nếu thuê máy vRAM thấp hơn (ví dụ RTX 3060 12GB), hãy giảm batch size trong `src/vtrace/depth_estimator.py` xuống `4` hoặc `2`.
- **Lỗi không nhận diện được PyTorch**: Hãy đảm bảo bạn đã chạy lệnh `source .venv/bin/activate` hoặc luôn chạy lệnh kèm tiền tố `uv run`.
