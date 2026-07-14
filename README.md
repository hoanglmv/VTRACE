# VTRACE - Viettel AI Race: Novel View Synthesis (3DGS)

Dự án này là mã nguồn tự động hoá việc huấn luyện (training) và sinh ảnh góc nhìn mới (inference) sử dụng phương pháp **3D Gaussian Splatting (MCMC Optimization với lõi gsplat)** dành riêng cho tập dữ liệu của cuộc thi Viettel AI Race (Digital Twin).

> **Profile chất lượng cao nhất:** pipeline chính cho lượt thuê GPU hiện là NVIDIA **3DGRUT + 3DGUT-MCMC + Neural Harmonic Textures**, được pin commit và tự resume theo scene/checkpoint. Xem [`NHT_MAX_RUNBOOK.md`](NHT_MAX_RUNBOOK.md) và [`config/nht_max.yaml`](config/nht_max.yaml). Pipeline gsplat cũ vẫn được giữ để đối chiếu, không còn là lựa chọn max-quality.

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

Profile NHT 4M yêu cầu GPU NVIDIA nhiều VRAM; quy trình chạy trên **Vast.ai** như sau:

#### 1. Thuê máy chủ GPU
- Khuyên dùng **A100/H100 80GB**; mức tối thiểu mà preflight chấp nhận là L40/L40S/A6000/RTX 6000 Ada 48GB.
- Chọn image Ubuntu có **CUDA 12.8 devel**, ít nhất 64GB RAM và 600GB persistent disk trống.

#### 2. Kết nối và Setup tự động
- Mở Terminal của máy chủ vừa thuê (hoặc thông qua Jupyter Lab Terminal).
- Kéo mã nguồn (git pull) và di chuyển vào thư mục VTRACE.
- Chạy script cài đặt tất cả trong một (DUY NHẤT 1 LỆNH CHẠY). Script này cài cả môi trường 3DGRUT/NHT max-quality đã pin commit:
  ```bash
  ./scripts/setup_all.sh
  ```
  *Lệnh này tự cài `uv`, môi trường repository, 3DGS legacy, môi trường tách biệt 3DGRUT/NHT đúng commit, kiểm tra CUDA và tải VAI_NVS_DATA khi dữ liệu chưa tồn tại.*

#### 3. Chạy Huấn luyện và Render Pipeline
Trước tiên chạy smoke test 10 iterations trên một public scene để buộc compile CUDA và xác nhận train/checkpoint/render:

```bash
./scripts/launch_nht_max.sh --smoke-test \
  --data-dir VAI_NVS_DATA/phase1/public_set --scene HCM0181
```

Khi `output_nht_smoke/DONE.json` xuất hiện, chạy toàn bộ **Private Set** bằng profile max-quality:

  ```bash
  ./scripts/launch_nht_max.sh
  ```

Launcher tách khỏi SSH bằng `setsid`/`nohup`; chạy lại đúng lệnh sẽ tự bỏ qua scene đã xong và resume scene bị ngắt từ checkpoint hợp lệ mới nhất.

#### 4. Tải kết quả nộp bài
- Khi pipeline chạy xong, file nén nằm tại `output_nht_max_private/submission_nht_max.zip`; chỉ sử dụng khi `DONE.json` cũng tồn tại.
- Chỉ cần click chuột phải vào file này trong cột thư mục bên trái của Jupyter Lab và chọn **Download** để tải về máy cá nhân của bạn.

---

### B. Quy trình chạy thủ công trên Laptop (Không cần GPU - Để phát triển code)

Nếu bạn chỉ chỉnh sửa code trên laptop cá nhân (không có GPU):
1. **Thiết lập code:** Chạy `uv sync` để cài đặt môi trường ảo Python.
2. **Chỉnh sửa:** Thực hiện chỉnh sửa code trực tiếp trên laptop và đẩy lên GitHub (hoặc dùng `scp`/`rsync` để đồng bộ code lên Vast.ai).

---

## 🛠️ Xử lý sự cố thường gặp (Troubleshooting)

- **Lỗi `Found no NVIDIA driver on your system`**: Lỗi này xảy ra khi bạn cố tình chạy huấn luyện/render trực tiếp trên laptop cá nhân không có GPU NVIDIA. Hãy đảm bảo bạn chỉ chạy lệnh chạy huấn luyện trên server Vast.ai.
- **Preflight báo thiếu VRAM/disk**: Không hạ ngầm profile 4M. Hãy đổi sang GPU 48/80GB hoặc tăng persistent disk theo thông báo.
- **Lỗi không nhận diện được PyTorch**: Hãy đảm bảo bạn đã chạy lệnh `source .venv/bin/activate` hoặc luôn chạy lệnh kèm tiền tố `uv run`.
