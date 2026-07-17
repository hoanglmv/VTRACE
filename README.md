# VTRACE - Viettel AI Race: Novel View Synthesis (3DGS)

Dự án này là mã nguồn tự động hoá việc huấn luyện (training) và sinh ảnh góc nhìn mới (inference) sử dụng phương pháp **3D Gaussian Splatting (MCMC Optimization với lõi gsplat)** dành riêng cho tập dữ liệu của cuộc thi Viettel AI Race (Digital Twin).

> **Profile chất lượng cao nhất:** pipeline chính cho lượt thuê GPU hiện là NVIDIA **3DGRUT + 3DGUT-MCMC + Neural Harmonic Textures**, được pin commit và tự resume theo scene/checkpoint. Xem [`NHT_MAX_RUNBOOK.md`](NHT_MAX_RUNBOOK.md) và [`config/nht_max.yaml`](config/nht_max.yaml). Pipeline gsplat cũ vẫn được giữ để đối chiếu, không còn là lựa chọn max-quality.

## 🗂️ Cấu trúc dự án (Src-layout)

- `src/vtrace/`: Package chứa mã nguồn cốt lõi.
  - `data_utils.py`: Đọc cấu trúc thư mục COLMAP và xử lý dữ liệu đầu vào.
  - `trainer.py`: Gọi vòng lặp huấn luyện tối ưu hóa bằng MCMC Strategy.
  - `renderer.py`: Sinh ảnh dựa vào file `test_poses.csv` (Quaternion/Translation) và đóng gói submission.
- `scripts/`: Chứa các script hệ thống.
  - `setup_all.sh`: Setup một lệnh cho VTRACE, 3DGS legacy, 3DGRUT/NHT đã pin commit và dữ liệu.
  - `launch_nht_max.sh`: Khởi chạy tách khỏi SSH, quản lý log và cho phép resume.
  - `run_nht_max.py`: Orchestrator train/render/checkpoint/validation/đóng gói dưới 350MB.
- `notebooks/`: Chứa các Jupyter Notebooks giúp quá trình chạy trở nên trực quan.
  - `01_data_analysis_and_training.ipynb`: Phân tích dữ liệu & Train.
  - `02_inference_and_submission.ipynb`: Render ảnh test & nén thành `submission.zip`.
- `VAI_NVS_DATA_ROUND2/`: Thư mục lưu dữ liệu thi. Script setup sẽ tự động tải cho bạn.

---

## 🚀 Hướng dẫn chạy trên server

### 1. Thuê server

Profile [`config/nht_max.yaml`](config/nht_max.yaml) sử dụng 3DGUT-MCMC + NHT với 1M primitives, 30.000 iterations và 48 NHT features. Cấu hình server:

- GPU: RTX 3090 24GB đang rảnh; không dùng RTX 3060 12GB.
- RAM: tối thiểu 64GB.
- Persistent disk: tối thiểu 200GB, nên thuê 250GB.
- Hệ điều hành: Ubuntu với CUDA 12.8 devel.

Kiểm tra server trước khi setup:

```bash
nvidia-smi
nvcc --version
df -h .
```

Preflight của pipeline yêu cầu ít nhất 22GB VRAM tổng và 20GB VRAM đang trống.

### 2. Setup một lần

Sau khi clone hoặc upload repository lên server:

```bash
cd /duong-dan/VTRACE
chmod +x scripts/setup_all.sh scripts/launch_nht_max.sh
./scripts/setup_all.sh
```

`setup_all.sh` tự động cài `uv`, môi trường VTRACE, 3DGS legacy, môi trường riêng cho 3DGRUT/NHT đúng commit, kiểm tra CUDA và tải dữ liệu nếu chưa tồn tại. Setup thành công khi terminal hiển thị:

```text
Hệ thống VTRACE đã thiết lập thành công!
```

### 3. Chạy smoke test bắt buộc

Smoke test chạy 10 iterations trên một public scene để kiểm tra CUDA compilation, train, checkpoint, camera adapter, render và đóng gói ZIP:

```bash
./scripts/launch_nht_max.sh \
  --smoke-test \
  --data-dir VAI_NVS_DATA_ROUND2/phase1/public_set \
  --scene HCM0181
```

Theo dõi launcher và log train:

```bash
tail -f output_nht_smoke/launcher.log
tail -f output_nht_smoke/logs/HCM0181.train.log
```

Nhấn `Ctrl+C` chỉ thoát lệnh `tail`, không dừng quá trình train. Smoke test chỉ thành công khi tồn tại `DONE.json`:

```bash
cat output_nht_smoke/DONE.json
ls -lh output_nht_smoke/submission.zip
```

### 4. Chạy public set và đo PSNR

Nên hoàn thành bước này trước private để xác nhận chất lượng thực tế:

```bash
./scripts/launch_nht_max.sh \
  --data-dir VAI_NVS_DATA_ROUND2/phase1/public_set \
  --output-dir output_nht_max_public
```

Theo dõi:

```bash
tail -f output_nht_max_public/launcher.log
```

Khi public run hoàn tất, đo PSNR/SSIM:

```bash
uv run python scripts/evaluate_public.py \
  --data-dir VAI_NVS_DATA_ROUND2/phase1/public_set \
  --prediction-dir output_nht_max_public/submission \
  --output-dir output_nht_max_public/evaluation \
  --no-lpips

cat output_nht_max_public/evaluation/summary.json
```

### 5. Chạy private set

Config mặc định đã trỏ tới toàn bộ private set:

```bash
./scripts/launch_nht_max.sh
```

Theo dõi tiến độ:

```bash
tail -f output_nht_max_private/launcher.log
find output_nht_max_private/scenes -name status.json -print
```

Có thể xem log của một scene cụ thể, ví dụ:

```bash
tail -f output_nht_max_private/logs/HCM0249.train.log
```

### 6. Kiểm tra kết quả cuối

Chỉ sử dụng submission khi cả `DONE.json` và `submission.zip` đều tồn tại:

```bash
cat output_nht_max_private/DONE.json
cat output_nht_max_private/packaging.json
ls -lh output_nht_max_private/submission.zip
unzip -t output_nht_max_private/submission.zip
```

Pipeline bắt đầu từ ảnh PNG lossless và tự chọn JPEG quality cao nhất sao cho ZIP đo thực tế nằm dưới giới hạn 350MB, với target an toàn 345MB. `DONE.json` lưu dung lượng byte, JPEG quality, subsampling và SHA-256 của file cuối.

### 7. Resume sau khi server bị ngắt

Launcher chạy qua `setsid`/`nohup`, vì vậy mất kết nối SSH không làm dừng train. Nếu server reboot hoặc bị preempt, chạy lại đúng lệnh:

```bash
./scripts/launch_nht_max.sh
```

Pipeline sẽ:

- Bỏ qua scene đã hoàn tất.
- Kiểm tra và bỏ qua checkpoint ghi dở.
- Resume scene bị ngắt từ checkpoint hợp lệ mới nhất.
- Không render model giả và không nuốt lỗi subprocess.

Không xóa thư mục `output_nht_max_private`, vì model, checkpoint và trạng thái resume nằm trong đó.

## 💻 Phát triển trên máy không có GPU

Nếu chỉ chỉnh sửa hoặc chạy unit tests trên laptop:

```bash
uv sync
uv run python -m unittest discover -s tests -v
```

Không chạy train/render NHT khi máy không có NVIDIA GPU.

## 🛠️ Xử lý sự cố

- `Found no NVIDIA driver`: server chưa gắn GPU hoặc NVIDIA driver không hoạt động; kiểm tra `nvidia-smi`.
- Preflight thiếu VRAM: dừng các tiến trình đang chiếm GPU; RTX 3090 phải còn ít nhất 20GB VRAM trống.
- Preflight thiếu disk: tăng persistent disk hoặc giải phóng dung lượng; không xóa checkpoint của run đang hoạt động.
- Setup không tìm thấy CUDA phù hợp: dùng image CUDA 12.8 devel; không dùng image chỉ có CUDA runtime mà thiếu `nvcc`.
- Mất SSH: kết nối lại và kiểm tra `launcher.log`; không khởi tạo một output directory mới nếu muốn resume.
