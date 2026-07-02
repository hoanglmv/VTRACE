# VTRACE - Viettel AI Race: Novel View Synthesis (3DGS)

Dự án này là mã nguồn tự động hoá việc huấn luyện (training) và sinh ảnh góc nhìn mới (inference) sử dụng phương pháp **3D Gaussian Splatting** dành riêng cho tập dữ liệu của cuộc thi Viettel AI Race (Digital Twin).

## 🗂️ Cấu trúc dự án (Src-layout)

- `src/vtrace/`: Package chứa mã nguồn cốt lõi.
  - `data_utils.py`: Đọc cấu trúc thư mục COLMAP và xử lý dữ liệu đầu vào.
  - `trainer.py`: Gọi script training của 3D Gaussian Splatting.
  - `renderer.py`: Sinh ảnh dựa vào file `test_poses.csv` (Quaternion/Translation) và đóng gói submission.
- `scripts/`: Chứa các script hệ thống.
  - `setup_3dgs.py`: Clone repo 3DGS và cài đặt tự động bằng `uv sync`.
- `notebooks/`: Chứa các Jupyter Notebooks giúp quá trình chạy trở nên trực quan.
  - `01_data_analysis_and_training.ipynb`: Phân tích dữ liệu & Train.
  - `02_inference_and_submission.ipynb`: Render ảnh test & nén thành `submission.zip`.
- `VAI_NVS_DATA/`: (Yêu cầu) Bạn cần đặt dữ liệu gốc của BTC tại thư mục này.

---

## 🚀 Hướng dẫn cài đặt và sử dụng

### 1. Chuẩn bị dữ liệu
Đảm bảo bạn đã giải nén tập dữ liệu của Ban Tổ Chức vào thư mục gốc của project với tên `VAI_NVS_DATA`.
Cấu trúc chuẩn của một scene sẽ như sau: `VAI_NVS_DATA/phase1/public_set/HCM0181/train/sparse/0/...`

### 2. Thiết lập Môi trường (Environment Setup)
Dự án sử dụng trình quản lý package siêu tốc `uv`. Mở Terminal tại thư mục gốc của dự án và chạy:
```bash
python scripts/setup_3dgs.py
```
> **Lưu ý**: Lệnh trên sẽ tự động `git clone` mã nguồn 3D Gaussian Splatting từ inria và chạy `uv sync` để biên dịch C++/CUDA cho `diff-gaussian-rasterization`. Hãy đảm bảo máy tính của bạn đã được cài sẵn bộ **CUDA Toolkit** (khuyến nghị phiên bản >= 11.8).

### 3. Huấn luyện Mô hình (Training)
1. Mở file `notebooks/01_data_analysis_and_training.ipynb`.
2. Đảm bảo bạn đã chọn kernel Python của môi trường ảo (ví dụ: `Python 3.10 (.venv)`).
3. Chạy từng Cell trong Notebook để phân tích dữ liệu, sau đó chạy khối Huấn luyện. Mô hình sau khi huấn luyện sẽ tự động lưu trong thư mục `output/`.

### 4. Sinh ảnh và tạo file Submission (Inference)
1. Mở file `notebooks/02_inference_and_submission.ipynb`.
2. Chạy khối lệnh để hệ thống tự động tải model vừa được train, đọc các tọa độ pose cần sinh ảnh từ `test_poses.csv`, và render ảnh kết quả.
3. Chạy khối lệnh cuối cùng để hệ thống nén toàn bộ thư mục `submission/` ra file `submission.zip` đúng chuẩn nộp bài của BTC!

---

## 🛠️ Xử lý sự cố thường gặp (Troubleshooting)
- **Lỗi không cài được `diff-gaussian-rasterization`**: Hãy chắc chắn bạn đã cài đặt `nvcc` bằng cách gõ `nvcc --version` vào Terminal. Nếu chưa có, bạn cần cài đặt NVIDIA CUDA Toolkit.
- **Lỗi ImportError vtrace trong Notebook**: Đảm bảo bạn đang chọn đúng kernel (môi trường `.venv` mà `uv` vừa tạo ra). Trong VS Code, bạn có thể bấm vào góc phải phía trên của Notebook để đổi Kernel.
