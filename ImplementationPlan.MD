# VTRACE — Kế hoạch cải thiện chất lượng 3DGS và rendering

> Trạng thái: **Chờ phê duyệt — chưa triển khai**  
> Ngày lập: 2026-07-14  
> Phạm vi: train 3D Gaussian Splatting, camera/rendering, appearance và đánh giá  
> Mục tiêu chính: thu hẹp khoảng cách từ khoảng 17 dB trên kết quả thực tế/private và 20.897 dB trên baseline public cục bộ tới vùng cạnh tranh trên 25 dB, không đánh đổi tính hợp lệ của submission.

## 1. Kết luận nghiên cứu

Khoảng cách 8 dB từ 17 lên 25 tương đương việc phải giảm MSE khoảng 6.3 lần. Đây không phải khoảng cách có thể bù chỉ bằng sharpening, JPEG quality hoặc tinh chỉnh một vài hệ số loss. Trình tự đúng là:

1. Xác minh evaluator và loại trừ sai lệch camera/rendering.
2. Thiết lập lại baseline 3DGS sạch ở full resolution, đủ 30k iterations và không early-stop sai.
3. So sánh có kiểm soát Default 3DGS, AbsGS và 3DGS-MCMC.
4. Chỉ thêm frequency/depth/appearance khi ablation chứng minh có lợi.
5. Mở nhánh high-ceiling với 3DGUT + Neural Harmonic Textures nếu baseline ổn định vẫn chưa đạt vùng mục tiêu.

Không có một thuật toán đơn lẻ nào được giả định chắc chắn sẽ đưa PSNR vượt 25. Mỗi thay đổi phải vượt cổng đánh giá trên public set trước khi được đưa sang private set.

## 2. Hiện trạng đã kiểm chứng

### 2.1 Dữ liệu

- Public set có 5 scene, tổng 1,160 ảnh train và 290 ảnh test.
- Mỗi scene có 200–240 ảnh train và 50–60 ảnh test, kích thước test `1320x989`.
- Camera là `SIMPLE_RADIAL`; `k1` khoảng `0.00784–0.01001`.
- Intrinsics trong `test_poses.csv` khớp COLMAP tới sai số kiểm tra bằng 0.
- Principal point nằm đúng tâm ảnh; đây không phải nguồn lỗi hiện tại.
- Test pose thuộc cùng mô hình/quỹ đạo SfM. Sau khi chỉ giữ camera có ảnh train thật, khoảng cách test-to-nearest-train có median `0.41–0.48 m`, p95 `0.81–1.49 m` tùy scene.
- Độ sáng trung bình giữa các ảnh có biến thiên vừa phải; độ lệch chuẩn luminance mẫu khoảng `2.7–6.6/255`. Appearance compensation đáng thử nhưng chưa đủ bằng chứng để coi là nguyên nhân chính của khoảng cách 8 dB.

### 2.2 Baseline cục bộ hiện có

File `output/evaluation_baseline/summary.json` ghi:

| Scene | PSNR |
|---|---:|
| HCM0181 | 20.356 |
| HCM0193 | 21.715 |
| HCM0204 | 19.958 |
| hcm0031 | 21.378 |
| hcm0034 | 21.079 |
| **Trung bình scene** | **20.897** |

Baseline này chưa phải baseline chất lượng cao:

- `resolution=2` thay vì full resolution.
- `SH degree=2` thay vì 3.
- Các run dừng ở khoảng 9,100–10,600 iterations, dù đích cấu hình là 30,000.
- Render baseline được tạo trước các sửa đổi mới về native radial distortion, 3DGUT/UT và lựa chọn rasterizer.

Con số khoảng 17 dB do người dùng báo được coi là kết quả thực tế/private hoặc pipeline nộp cũ. Kế hoạch sẽ lưu riêng `public_local_psnr` và `leaderboard_psnr`; tuyệt đối không so sánh chéo hai nguồn như cùng một benchmark.

### 2.3 Mã hiện tại

- Backend đang khóa ở `gsplat==1.5.3`, có `DefaultStrategy`, `absgrad`, `MCMCStrategy`, antialiased rasterization, radial distortion và 3DGUT (`with_ut`, `with_eval3d`).
- Workspace có thay đổi chưa commit cho native distortion, exact intrinsics, AbsGS/MCMC, frequency loss, affine exposure interpolation, evaluator và render sweep.
- Các thay đổi này mới qua kiểm tra cú pháp/camera; chưa có full GPU ablation sau lần sửa cuối.
- Current `rasterize_mode="antialiased"` chỉ cung cấp phần 2D opacity compensation lấy cảm hứng từ Mip-Splatting; nó **không tương đương toàn bộ Mip-Splatting**, vì chưa có 3D smoothing filter.
- Frequency loss hiện tại là một biến thể đơn giản hóa. Không được gọi là tái hiện FreGS chính xác trước khi đối chiếu công thức và ablation.
- MCMC hiện chưa dùng đúng cặp regularizer opacity/scale theo reference implementation. `lambda_scale` hiện tại là thresholded scale penalty khác với regularizer của 3DGS-MCMC.
- Dense initialization hiện tạo khoảng 7.1 triệu điểm/scene. Đây là mức rất lớn cho RTX 3090/4090, làm tăng VRAM và thời gian trước khi densification; không nên bật mặc định nếu chưa chứng minh tăng validation PSNR.

## 3. Nghiên cứu phương pháp và quyết định áp dụng

| Phương pháp/framework | Giá trị đối với VTRACE | Quyết định |
|---|---|---|
| Native `gsplat` + 3DGUT | Hỗ trợ radial distortion trực tiếp bằng Unscented Transform; rất khớp camera `SIMPLE_RADIAL` | **P0 — dùng ngay và kiểm chứng train/render parity** |
| AbsGS | Giảm gradient collision, thúc đẩy split ở vùng chi tiết bị blur | **P1 — benchmark chính** |
| 3DGS-MCMC | Chất lượng mạnh, kiểm soát số Gaussian, ít phụ thuộc initialization hơn | **P1 — benchmark chính, sửa regularizer đúng paper** |
| Mip-Splatting | 3D smoothing + 2D mip filter để chống aliasing | **P1/P2 — trước tiên ablate 2D mode có sẵn; sau đó mới port 3D filter** |
| FreGS | Progressive frequency regularization cho vùng high-frequency bị over-reconstruction | **P2 — triển khai đúng paper sau khi baseline sạch** |
| Pixel-GS | Hữu ích khi sparse initialization làm vùng lớn không được densify | **P2 — chỉ thử nếu diagnostics cho thấy thiếu Gaussian theo vùng** |
| Bilateral grid / affine exposure | Bù ISP/appearance giữa ảnh train | **P2 — đánh giá leave-one-out; không mặc định áp lên test** |
| PPISP (CVPR 2026 Oral) | Mô hình ISP có cơ sở vật lý và controller cho novel views; phù hợp hơn nội suy affine đơn giản | **P3 — high-ceiling appearance track** |
| Neural Harmonic Textures (ECCV 2026) | Appearance model mới gắn vào 3DGUT, đạt kết quả mạnh trên NVS và chi tiết tần số cao | **P3 — nhánh high-ceiling riêng** |
| 3DGRUT framework | Reference chính thức cho 3DGUT, MCMC, PPISP/NHT; có UV setup | **P3 — adapter riêng, không thay nóng pipeline đang chạy** |
| WildGaussians/SWAG | Mạnh cho ảnh in-the-wild, transient và thay đổi illumination lớn | Giữ tham khảo; dữ liệu VTRACE có quỹ đạo/ISP khá đồng nhất nên không ưu tiên trước PPISP |
| StopThePop | Tăng nhất quán theo chuyển động, nhưng paper báo chất lượng ảnh tĩnh tương đương | Không ưu tiên cho PSNR leaderboard |
| 2DGS/GOF/SuGaR | Nhấn mạnh geometry/surface reconstruction | Không ưu tiên khi metric chính là image NVS |
| Few-shot/diffusion NVS | Nhắm 2–nhiều rất ít view, trong khi VTRACE có 200–240 view | Loại khỏi đường chính để tránh hallucination làm giảm PSNR |

Nguồn nghiên cứu chính:

- [gsplat rasterization API: antialiasing, AbsGrad và radial distortion/3DGUT](https://docs.gsplat.studio/main/apis/rasterization.html)
- [gsplat repository và các cập nhật 2026](https://github.com/nerfstudio-project/gsplat)
- [3DGRT/3DGUT official implementation](https://github.com/nv-tlabs/3dgrut)
- [AbsGS: Recovering Fine Details for 3D Gaussian Splatting](https://arxiv.org/abs/2404.10484)
- [3D Gaussian Splatting as Markov Chain Monte Carlo](https://github.com/ubc-vision/3dgs-mcmc)
- [Mip-Splatting official implementation](https://github.com/autonomousvision/mip-splatting)
- [FreGS, CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Zhang_FreGS_3D_Gaussian_Splatting_with_Progressive_Frequency_Regularization_CVPR_2024_paper.html)
- [Pixel-GS](https://pixelgs.github.io/)
- [Nerfstudio Splatfacto reference: MCMC regularizers và bilateral grid](https://github.com/nerfstudio-project/nerfstudio/blob/main/nerfstudio/models/splatfacto.py)
- [PPISP, CVPR 2026 Oral](https://research.nvidia.com/labs/sil/projects/ppisp/)
- [Neural Harmonic Textures, ECCV 2026](https://research.nvidia.com/labs/sil/projects/neural-harmonic-textures/)
- [WildGaussians official code](https://github.com/jkulhanek/wild-gaussians/)
- [StopThePop official code](https://github.com/r4dl/StopThePop)

## 4. Giả thuyết nguyên nhân — theo thứ tự kiểm tra

### H1 — Baseline đang under-train

Độ tin cậy: **rất cao**.

Evidence: half resolution, SH2 và early-stop ở khoảng 1/3 lịch train. Đây có thể giải thích một phần lớn khoảng cách từ 20.9 đến vùng giữa 20 dB.

### H2 — Train/render camera model chưa hoàn toàn đồng nhất

Độ tin cậy: **cao**.

Native 3DGUT, legacy OpenCV remap và pinhole/no-distortion đang cùng tồn tại. Chỉ cần lệch vài pixel ở biên ảnh đã gây tổn thất PSNR lớn. Phải dùng cùng intrinsics, distortion convention, resolution scaling và projection trong train lẫn render.

### H3 — MCMC integration/regularization chưa đúng reference

Độ tin cậy: **cao**.

Current code tự bridge một optimizer nhiều param-group sang strategy API và dùng regularizer khác paper. Phải có unit/integration tests cho số Gaussian, optimizer state và gradient trước khi tin metric.

### H4 — Densification chưa phân bổ đủ chi tiết

Độ tin cậy: **trung bình-cao**.

Ảnh drone có mái nhà, đường, cây và texture tần số cao. AbsGS/FreGS/Pixel-GS có cơ sở phù hợp. Tuy nhiên không được bật đồng thời ngay từ đầu vì sẽ không biết thành phần nào có tác dụng.

### H5 — ISP/appearance variation bị bake vào geometry

Độ tin cậy: **trung bình**.

Luminance có biến thiên nhưng không cực đoan. Nội suy exposure theo pose/frame hoặc PPISP có thể cải thiện, song phải đánh giá bằng leave-one-out vì test không có appearance code quan sát được.

### H6 — Initialization/depth chưa phù hợp

Độ tin cậy: **trung bình-thấp** cho toàn bộ dataset.

COLMAP đã có khoảng 200k–250k sparse points và nhiều view. Dense Depth Anything V2 initialization 7M điểm có thể tốn tài nguyên và đưa depth bias vào mô hình. Chỉ giữ làm ablation trên scene xấu nhất.

## 5. Mục tiêu và cổng chất lượng

### Metric

- Metric quyết định chính: mean PSNR theo scene trên public test GT.
- Metric bảo vệ: SSIM, LPIPS, số Gaussian, VRAM peak, thời gian train/render và validation theo vùng ảnh.
- Evaluator cục bộ hiện giả định score tổng hợp `0.4*(1-LPIPS) + 0.3*SSIM + 0.3*PSNR_norm`. Vì chưa có văn bản thể lệ chính thức trong phiên hiện tại, công thức này phải được xác nhận ở Phase 0; trước khi xác nhận, nó chỉ là giả thuyết của repository.

### Cổng quyết định

- **G0 — Evaluator parity:** tên file, RGB range, resize/crop, PSNR/SSIM/LPIPS và averaging khớp thể lệ chính thức.
- **G1 — Camera parity:** render lại train views có projection/distortion khớp; không có sai số hệ thống tăng theo bán kính ảnh.
- **G2 — Baseline sạch:** full-resolution 30k run hoàn tất, không early-stop, reproducible giữa hai seed trong sai số hợp lý.
- **G3 — Candidate:** thay đổi phải tăng ít nhất `+0.20 dB` mean PSNR trên hai scene development, không làm scene nào giảm quá `0.30 dB`, hoặc cải thiện rõ score chính thức nếu PSNR giảm rất nhỏ.
- **G4 — Promotion:** candidate thắng trên cả 5 public scenes và không dựa vào test GT trong train.
- **G5 — Private:** chỉ chạy private bằng config đã đóng băng từ public; không tuning theo phản hồi leaderboard quá mức.

Mốc tham vọng, không phải cam kết:

1. Baseline sạch: vượt rõ 20.897 dB public hiện tại.
2. Pipeline ổn định: tiến tới `>=23 dB` public trước khi mở nhánh framework lớn.
3. Candidate cạnh tranh: tiến tới vùng `24–25+ dB` public và kiểm tra tương quan với private/leaderboard.

Nếu baseline sạch vẫn dưới 22 dB, dừng tuning loss và quay lại audit camera/data. Nếu public vượt 24 dB nhưng private vẫn quanh 17 dB, ưu tiên điều tra domain/packaging/evaluator mismatch thay vì tiếp tục thêm thuật toán.

## 6. Thiết kế validation không rò rỉ dữ liệu

- Public test GT chỉ dùng để đánh giá và chọn cấu hình tổng quát, không đưa vào loss.
- Tạo thêm protocol train-view holdout: giữ lại mỗi ảnh thứ 5 theo frame index hoặc chia block theo quỹ đạo, sao cho mô phỏng đúng test interleaving.
- Với appearance controller/interpolation, dùng leave-one-out: học appearance từ các frame còn lại rồi dự đoán frame bị giữ lại.
- Báo cả mean theo ảnh và macro mean theo scene; quyết định promotion dựa trên macro mean để tránh scene 60 ảnh lấn scene 50 ảnh.
- Chạy development trước trên `HCM0204` (baseline thấp, có test pose xa nhất) và `HCM0193` (baseline cao, quỹ đạo phủ tốt), sau đó mới xác nhận đủ 5 scene.

## 7. Kế hoạch triển khai theo phase

## Phase 0 — Đóng băng baseline và evaluator

Mục tiêu: có một phép đo đáng tin cậy trước khi sửa thuật toán.

- [ ] Xin/đọc lại thể lệ chính thức và xác nhận công thức metric, LPIPS backbone, PSNR cap, color space, crop/border và quy tắc averaging.
- [ ] Tạo `experiments/baseline_legacy/manifest.yaml` ghi commit, dirty diff hash, Python/Torch/CUDA/gsplat, GPU và config resolved.
- [ ] Kiểm tra submission strict: đủ scene, đúng tên/extension/kích thước, không file thừa.
- [ ] Thêm evaluator partial-scene phục vụ development nhưng không cho phép dùng partial result làm final score.
- [ ] Lưu heatmap/error theo radial bins, center/edge và image index để phát hiện distortion/temporal drift.
- [ ] Tách rõ `public_local_psnr` và `leaderboard_psnr` trong mọi report.

Deliverable: `reports/phase0_baseline.md`, metrics CSV/JSON và manifest tái lập được.

## Phase 1 — Camera và render-only sweep

Mục tiêu: sửa lỗi hệ thống với chi phí thấp, không retrain.

- [ ] Viết round-trip tests cho COLMAP quaternion/translation, camera center, K scaling và radial projection.
- [ ] Bảo đảm train và inference đều dùng exact `fx, fy, cx, cy`, cùng scale convention.
- [ ] So sánh các projection từ cùng checkpoint:
  - native radial + `with_ut=true`, `with_eval3d=false`;
  - native radial + `with_ut=true`, `with_eval3d=true`;
  - legacy OpenCV remap;
  - no distortion chỉ làm negative control.
- [ ] So sánh `classic` và `antialiased` rasterization.
- [ ] Sweep supersampling `1.0, 1.5, 2.0` với area/lanczos; loại sớm variant giảm PSNR.
- [ ] Sweep JPEG `95, 97, 100`, chroma subsampling `0/2`, và PNG nếu luật cho phép giữ đúng tên; chọn bằng public metrics, không chọn bằng cảm quan.
- [ ] Không dùng sharpening/post-processing mặc định. Chỉ sweep rất nhỏ sau khi projection đúng.

Ma trận render đầu tiên dùng checkpoint baseline hiện có để rẻ. Variant thắng sẽ được xác nhận lại trên checkpoint full-resolution ở Phase 2.

Deliverable: `reports/render_ablation.csv`, ảnh difference và một render profile được đóng băng.

## Phase 2 — Baseline 3DGS sạch, full resolution

Mục tiêu: tách cải thiện do train đủ khỏi cải thiện do thuật toán mới.

- [ ] Tắt early stopping cho benchmark; luôn lưu checkpoint `7k, 15k, 20k, 30k`.
- [ ] Full resolution (`resolution=1`), SH3, 30k, COLMAP sparse initialization.
- [ ] Dùng native distortion/3DGUT đồng nhất trong train và render.
- [ ] Bắt đầu bằng loss chuẩn `L1 + 0.2*DSSIM`; tắt edge, depth, frequency, exposure và custom pruner.
- [ ] Tắt dense Depth Anything initialization ở baseline.
- [ ] Cố định seed, ghi peak VRAM, số Gaussian và metric từng checkpoint.
- [ ] Chạy hai seed trên một scene để đo variance; sau đó một seed trên cặp development.
- [ ] Đối chiếu một run với reference `gsplat` simple trainer hoặc Nerfstudio Splatfacto để phát hiện sai lệch integration.

Candidate `B0`: original/default densification.  
Candidate `B1`: AbsGS (`absgrad=true`) với threshold paper/reference.  
Candidate `B2`: MCMC với `cap_max` được chọn theo VRAM và regularizer đúng reference.

Không thêm FreGS vào `B1`; file `public_absgs_fregs.yaml` hiện tại sẽ được đổi tên/đóng băng như experimental, tránh nhầm nó là baseline.

Deliverable: bảng Default vs AbsGS vs MCMC theo checkpoint và scene.

## Phase 3 — Densification và loss ablation

Mục tiêu: tăng chi tiết mà không overfit train views.

Chỉ bắt đầu từ winner của Phase 2.

### 3A. Densification

- [ ] AbsGS: sweep `grow_grad2d`, revised opacity và refine stop iteration.
- [ ] MCMC: sweep `cap_max`, opacity regularizer, scale regularizer và noise schedule; giữ công thức reference trước khi thử biến thể annealing.
- [ ] Kiểm tra bridge optimizer khi strategy clone/split/relocate: shape, state Adam và parameter identity phải đồng bộ.
- [ ] Chỉ port Pixel-GS nếu radial error map cho thấy vùng thiếu initialization/densification.

### 3B. Anti-aliasing

- [ ] Ablate gsplat 2D antialiased mode trong cả train và render; không trộn classic-train với antialiased-render nếu chưa đo.
- [ ] Nếu có aliasing rõ và 2D mode thắng, port thêm 3D smoothing filter của Mip-Splatting như feature flag riêng.
- [ ] Không dùng supersampling như thay thế cho train-time frequency control; nó chỉ là render ablation.

### 3C. Loss

- [ ] Sweep DSSIM nhỏ: `0.2, 0.3, 0.4`.
- [ ] Thay edge-weight hiện tại bằng ablation có normalization để tổng learning-rate hiệu dụng không đổi.
- [ ] Tái hiện FreGS theo paper: low/high-frequency decomposition, progressive schedule và loss weighting; so với simplified FFT loss hiện có.
- [ ] Không optimize LPIPS trực tiếp ở đường đầu vì có thể làm giảm PSNR và tăng VRAM; chỉ thử sau khi metric tổng hợp được xác nhận.

### 3D. Depth/initialization

- [ ] So sánh sparse COLMAP, filtered dense initialization và depth regularization chỉ trên `HCM0204`.
- [ ] Giảm dense seed từ ~7.1M xuống một ngân sách có kiểm soát bằng confidence/voxel/visibility; không đưa 7M điểm làm mặc định.
- [ ] Thay Pearson-only depth loss bằng scale/shift-invariant depth supervision đã được mask nếu ablation cho thấy geometry là bottleneck.
- [ ] Tắt depth floater pruner khi không có test chứng minh precision; current multi-camera voting có thể xóa Gaussian hợp lệ.

Deliverable: một bảng one-factor-at-a-time, không gộp nhiều thay đổi vào cùng experiment ID.

## Phase 4 — Appearance và render theo quỹ đạo

Mục tiêu: xử lý exposure/white balance và phục hồi texture gần camera test.

### 4A. Appearance an toàn

- [ ] Sửa/làm rõ per-image affine transform là ma trận `3x4`, không dùng nhầm exponential scalar exposure của code cũ.
- [ ] Regularize transform về identity và log condition number để tránh model “ăn gian” màu.
- [ ] Dự đoán test appearance bằng pose + frame interpolation, đánh giá leave-one-out trên train và public test.
- [ ] So sánh affine interpolation với bilateral grid của Splatfacto.
- [ ] Nếu lợi ích ổn định, thử PPISP controller trong nhánh riêng; không port toàn bộ trước khi affine/bilateral baseline có số liệu.

### 4B. Hybrid 3DGS + image-based reprojection

Đây là optional render track nhưng phù hợp dữ liệu vì test view gần train view.

- [ ] Render RGB, depth và alpha từ 3DGS.
- [ ] Chọn 2–4 train frame gần nhất theo camera center + hướng nhìn + frame index.
- [ ] Reproject train RGB qua depth, kiểm tra z-buffer/occlusion và forward-backward consistency.
- [ ] Blend residual high-frequency vào 3DGS chỉ tại pixel confidence cao; 3DGS giữ vai trò fallback ở disocclusion.
- [ ] Tune toàn bộ confidence/blend bằng leave-one-out, không nhìn private GT.
- [ ] Loại hướng này nếu tăng PSNR nhưng làm LPIPS/SSIM hoặc scene robustness giảm vượt cổng G3.

## Phase 5 — High-ceiling framework track

Chỉ mở nếu winner Phase 3/4 chưa vào vùng mục tiêu hoặc nếu GPU budget cho phép.

### 5A. 3DGRUT reference

- [ ] Tạo adapter chuyển VTRACE COLMAP + `test_poses.csv` sang config của 3DGRUT, giữ nguyên coordinate convention và radial distortion.
- [ ] Cài trong environment riêng, pin commit; không thay dependency của pipeline chính.
- [ ] Reproduce 3DGUT-MCMC baseline trên `HCM0204` và `HCM0193`.
- [ ] So với gsplat pipeline bằng cùng train/test manifest và evaluator.

### 5B. Neural Harmonic Textures

- [ ] Bật NHT trên 3DGUT với ngân sách 2M rồi tối đa 5M primitives nếu VRAM cho phép.
- [ ] Bắt đầu theo cấu hình official, không sweep MLP/features đồng thời với geometry.
- [ ] Đo PSNR/SSIM/LPIPS, VRAM, train time và render time.
- [ ] Promotion chỉ khi tăng ổn định trên cả hai development scenes; sau đó mới chạy đủ public/private.

NHT là hướng có trần chất lượng cao nhất trong nghiên cứu hiện tại: trang official báo `28.74/0.834/0.216` trên Mip-NeRF360 và `25.68/0.882/0.141` trên Tanks & Temples, cao hơn các baseline primitive trong bảng của họ. Đây là bằng chứng để ưu tiên thử, **không phải dự báo trực tiếp cho VTRACE**.

## Phase 6 — Xác nhận full public, private và đóng gói

- [ ] Khóa config trước khi chạy private.
- [ ] Chạy đủ 5 public scenes với cùng seed policy và winner pipeline.
- [ ] Chạy đủ private scenes, resume theo scene/checkpoint.
- [ ] Validate 100% manifest, dimensions, filenames, image decode và ZIP root structure.
- [ ] Lưu checksum từng ảnh và ZIP.
- [ ] Sinh model card/report gồm commit, config, dependency lock, GPU, runtime, metrics và known risks.
- [ ] Giữ tối thiểu một submission conservative và một submission aggressive; không trộn ảnh per-scene nếu luật không cho phép hoặc chưa có public evidence.

## 8. Ma trận experiment tối thiểu

| ID | Train | Render | Scene dev | Mục đích |
|---|---|---|---|---|
| R0 | checkpoint cũ | legacy remap/classic | 2 scene | Neo baseline render |
| R1 | checkpoint cũ | native UT/classic | 2 scene | Distortion parity |
| R2 | checkpoint cũ | native UT/antialiased | 2 scene | 2D mip compensation |
| R3 | checkpoint cũ | native UT/eval3d | 2 scene | 3D response accuracy |
| B0 | Default, full-res, clean loss | winner R* | 2 scene | Clean baseline |
| B1 | AbsGS, full-res | winner R* | 2 scene | Fine-detail densification |
| B2 | MCMC reference regs | winner R* | 2 scene | MCMC baseline đúng |
| A1 | winner B* + FreGS | winner R* | 2 scene | Frequency detail |
| A2 | winner B* + exposure/appearance | winner R* | 2 scene | ISP variation |
| A3 | winner B* + controlled depth | winner R* | HCM0204 | Geometry bottleneck |
| H1 | winner B* | hybrid reprojection | 2 scene | Texture recovery |
| N1 | 3DGRUT 3DGUT-MCMC | official | 2 scene | External reference |
| N2 | 3DGRUT + NHT | official | 2 scene | High-ceiling candidate |

Mỗi experiment thay đúng một yếu tố so với parent. Chỉ winner qua G3 mới được mở rộng từ 2 lên 5 public scenes.

## 9. Cơ chế chống mất tiến độ/cancel

Mọi run sau khi được duyệt phải idempotent và resume được theo scene.

### Cấu trúc output đề xuất

```text
experiments/<experiment_id>/
├── manifest.yaml
├── config.resolved.yaml
├── status.json
├── environment.txt
├── git.diff.patch
├── scenes/
│   └── <scene>/
│       ├── status.json
│       ├── train.log
│       ├── checkpoints/
│       ├── renders/
│       ├── metrics.json
│       └── DONE
├── metrics_per_scene.csv
└── summary.json
```

### Quy tắc resume

- `status.json` ghi `pending/running/completed/failed`, PID/host, timestamps và bước cuối.
- Ghi status bằng temp file + atomic rename để tránh JSON hỏng khi process bị kill.
- Scene có marker `DONE` và checksum hợp lệ sẽ được skip.
- Checkpoint định kỳ và loss history được flush mỗi 1,000 iterations.
- Render theo từng ảnh, atomic rename từ `.tmp`; resume bỏ qua ảnh đã decode đúng kích thước/checksum.
- Không ghi đè experiment cũ. Config thay đổi tạo experiment ID mới.
- Mỗi run lưu dirty diff để tái dựng ngay cả khi thay đổi chưa commit.
- Có lệnh `--resume`, `--scene`, `--from-stage`, `--dry-run` và `--validate-only`.
- Job dài chạy qua `tmux`/`systemd-run`/scheduler trên GPU server; log vẫn được stream vào file. Công cụ orchestration không phải nguồn trạng thái duy nhất.

## 10. Thay đổi mã dự kiến sau phê duyệt

Các file/module dự kiến, tên có thể điều chỉnh khi triển khai:

- `src/vtrace/camera_model.py`: conversion, projection và distortion tests dùng chung train/render.
- `src/vtrace/experiment.py`: manifest, status, resume và atomic outputs.
- `src/vtrace/evaluator.py`: official metric parity, partial dev evaluation và error diagnostics.
- `src/vtrace/patches/train.py`: clean strategy adapter, correct MCMC regularizers, feature flags.
- `src/vtrace/patches/gaussian_renderer.py`: renderer profile rõ ràng, 3DGUT/eval3d và Mip features.
- `src/vtrace/appearance.py`: leave-one-out appearance predictor/controller.
- `src/vtrace/hybrid_renderer.py`: optional depth-guided reprojection.
- `pipeline/run_pipeline.py`: stage/scene resume, resolved config và strict promotion.
- `scripts/run_experiment.py`: entry point thống nhất.
- `scripts/render_sweep.py`: render-only ablation có manifest.
- `scripts/evaluate_public.py`: official/strict mode.
- `tests/`: camera, strategy state, metrics và submission tests.
- `config/experiments/`: immutable experiment configs; không sửa config đã chạy.

## 11. Rủi ro và rollback

- **Metric mismatch:** không tối ưu tiếp trước khi G0 đạt.
- **Camera convention:** giữ legacy renderer làm oracle/rollback; mọi projection change có golden tests.
- **OOM:** cap Gaussian, sparse/packed mode, giảm feature budget; không tự động hạ resolution vì sẽ làm experiment không còn tương đương.
- **Dependency main branch:** gsplat main/3DGRUT/NHT luôn pin commit trong env riêng; pipeline stable vẫn giữ được.
- **Dirty worktree:** snapshot patch trước triển khai; không reset/xóa thay đổi hiện có.
- **Overfit public:** config được chọn theo hai-scene development + full-public confirmation, sau đó khóa trước private.
- **Appearance leakage:** mọi predictor phải vượt leave-one-out mà không dùng target image.
- **Post-processing hallucination:** chỉ dùng deterministic, confidence-gated operations; không diffusion/generative enhancement trong đường PSNR chính.

## 12. Thứ tự ưu tiên và ngân sách

Ưu tiên theo lợi ích/rủi ro:

1. **P0:** evaluator, full-res/30k, tắt early-stop, camera/render parity.
2. **P1:** Default vs AbsGS vs MCMC reference; native UT/eval3d/antialias ablation.
3. **P2:** FreGS chính xác, controlled Mip 3D filter, appearance leave-one-out, depth ablation.
4. **P3:** hybrid reprojection, PPISP, 3DGRUT + NHT.

GPU budget được cấp theo cổng:

- Render sweep: dùng checkpoint cũ, chi phí thấp.
- Phase 2: chỉ 2 development scenes trước.
- Full 5-scene run chỉ dành cho winner qua G3.
- NHT/PPISP chạy environment riêng và chỉ sau khi baseline vượt G1/G2.

## 13. Nội dung cần người dùng phê duyệt

Đề xuất phê duyệt theo mặc định:

1. Cho phép bắt đầu **Phase 0 → Phase 2**, chưa mở NHT/PPISP.
2. Giữ `gsplat==1.5.3` cho baseline; chỉ pin gsplat main/3DGRUT trong environment experimental riêng.
3. Dùng `HCM0204` và `HCM0193` làm development scenes.
4. Tắt early stopping cho mọi benchmark 30k.
5. Không bật dense Depth Anything initialization, custom edge/frequency loss hoặc post-processing trong baseline sạch.
6. Không xóa hay reset các thay đổi chưa commit; snapshot chúng trước khi triển khai.

Sau khi Phase 2 có bảng kết quả, người dùng duyệt lần hai để chọn một trong ba nhánh: tiếp tục AbsGS/MCMC tuning, mở appearance/hybrid rendering, hoặc chuyển sang 3DGRUT + NHT.
