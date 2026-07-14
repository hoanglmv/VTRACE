import sys
import os
import csv
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import logging
import cv2

logger = logging.getLogger(__name__)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "gaussian-splatting")))

try:
    from scene.gaussian_model import GaussianModel
    from gaussian_renderer import render
    from utils.graphics_utils import fov2focal, focal2fov
    from scene.cameras import Camera
    from scene.colmap_loader import read_intrinsics_binary, read_intrinsics_text
except ImportError:
    logger.warning("3DGS modules not found. Ensure setup.py has been run and gaussian-splatting is cloned.")

class PipelineParams:
    def __init__(self, antialiasing=False, with_ut=True, with_eval3d=False):
        self.compute_cov3D_python = False
        self.convert_SHs_python = False
        self.debug = False
        self.antialiasing = antialiasing
        self.with_ut = with_ut
        self.with_eval3d = with_eval3d

def qvec2rotmat(qvec):
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])

def generate_dummy_renders(test_csv, out_scene_dir, render_format):
    os.makedirs(out_scene_dir, exist_ok=True)
    fmt = render_format.strip(".").lower()
    if fmt == "jpg":
        fmt = "jpeg"
    ext = "." + fmt
    if ext not in [".png", ".jpeg"]:
        ext = ".png"
        fmt = "png"
    
    with open(test_csv, "r") as f:
        reader = csv.DictReader(f)
        for row in tqdm(list(reader), desc=f"Generating dummy renders"):
            img_name = row["image_name"]
            width, height = int(row["width"]), int(row["height"])
            dummy = Image.new("RGB", (width, height), (0, 0, 0))
            
            ext_in_csv = os.path.splitext(img_name)[1]
            if ext_in_csv:
                out_img_name = img_name
                save_fmt = "JPEG" if ext_in_csv.lower() in [".jpg", ".jpeg"] else "PNG"
            else:
                out_img_name = img_name + ext
                save_fmt = "JPEG" if fmt == "jpeg" else "PNG"
                
            out_path = os.path.join(out_scene_dir, out_img_name)
            if save_fmt == "JPEG":
                dummy.save(out_path, "JPEG", quality=100)
            else:
                dummy.save(out_path, "PNG")

def render_scene(
    scene_name,
    scene_dir,
    model_path,
    output_dir,
    render_format="png",
    *,
    antialiasing=False,
    distortion_mode="native",
    with_ut=True,
    with_eval3d=False,
    supersample=1.0,
    downsample_filter="area",
    jpeg_quality=100,
    checkpoint_iteration=None,
):
    test_csv = os.path.join(scene_dir, "test", "test_poses.csv")
    if not os.path.exists(test_csv):
        logger.warning(f"No test_poses.csv found for {scene_name}")
        return

    # Determine SH degree from PLY file dynamically
    import math
    from plyfile import PlyData
    
    out_scene_dir = os.path.join(output_dir, scene_name)
    iter_dir = os.path.join(model_path, "point_cloud")
    if not os.path.exists(iter_dir):
        logger.warning(f"Model not found at {iter_dir}, falling back to dummy renders.")
        generate_dummy_renders(test_csv, out_scene_dir, render_format)
        return
        
    iterations = sorted([int(i.split("_")[-1]) for i in os.listdir(iter_dir) if "iteration_" in i])
    if not iterations:
        logger.warning("No iterations found, falling back to dummy renders.")
        generate_dummy_renders(test_csv, out_scene_dir, render_format)
        return
        
    if checkpoint_iteration is None or str(checkpoint_iteration).lower() == "latest":
        latest_iter = iterations[-1]
    else:
        latest_iter = int(checkpoint_iteration)
        if latest_iter not in iterations:
            raise ValueError(
                f"Requested checkpoint iteration {latest_iter} is unavailable for {scene_name}; "
                f"available iterations: {iterations}"
            )
    ply_path = os.path.join(iter_dir, f"iteration_{latest_iter}", "point_cloud.ply")
    
    try:
        plydata = PlyData.read(ply_path)
        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        detected_sh_degree = int(math.sqrt(len(extra_f_names) // 3 + 1)) - 1
        logger.info(f"Detected SH degree {detected_sh_degree} from PLY file {ply_path}")
    except Exception as e:
        logger.error(f"Error reading PLY header: {e}")
        detected_sh_degree = 3
        
    gaussians = GaussianModel(sh_degree=detected_sh_degree)
    gaussians.load_ply(ply_path)
    
    background = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
    pipeline = PipelineParams(
        antialiasing=antialiasing,
        with_ut=with_ut,
        with_eval3d=with_eval3d,
    )
    if distortion_mode not in {"native", "legacy_remap", "none"}:
        raise ValueError(
            f"Unknown distortion_mode={distortion_mode!r}; expected native, legacy_remap, or none"
        )
    if supersample < 1.0:
        raise ValueError("supersample must be >= 1.0")
    
    # Load radial distortion parameters from training sparse reconstruction
    radial_coeffs = np.zeros(6, dtype=np.float32)
    sparse_dir = os.path.join(scene_dir, "train", "sparse", "0")
    cameras_bin = os.path.join(sparse_dir, "cameras.bin")
    cameras_txt = os.path.join(sparse_dir, "cameras.txt")
    try:
        if os.path.exists(cameras_bin):
            cameras_data = read_intrinsics_binary(cameras_bin)
        elif os.path.exists(cameras_txt):
            cameras_data = read_intrinsics_text(cameras_txt)
        else:
            cameras_data = {}
            
        if cameras_data:
            cam_colmap = list(cameras_data.values())[0]
            if cam_colmap.model in ["SIMPLE_RADIAL", "RADIAL"]:
                k1 = cam_colmap.params[3] if len(cam_colmap.params) > 3 else 0.0
                k2 = cam_colmap.params[4] if len(cam_colmap.params) > 4 else 0.0
                radial_coeffs = np.array([k1, k2, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
            elif cam_colmap.model in ["OPENCV", "OPENCV_FISHEYE"]:
                k1 = cam_colmap.params[4] if len(cam_colmap.params) > 4 else 0.0
                k2 = cam_colmap.params[5] if len(cam_colmap.params) > 5 else 0.0
                radial_coeffs = np.array([k1, k2, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    except Exception as e:
        logger.warning(f"Could not load radial distortion from training folder: {e}. Defaulting to zero distortion.")
    
    out_scene_dir = os.path.join(output_dir, scene_name)
    os.makedirs(out_scene_dir, exist_ok=True)
    
    # Determine extension and format
    fmt = render_format.strip(".").lower()
    if fmt == "jpg":
        fmt = "jpeg"
    ext = "." + fmt
    if ext not in [".png", ".jpeg"]:
        ext = ".png"
        fmt = "png"
        
    with open(test_csv, "r") as f:
        reader = csv.DictReader(f)
        
        for idx, row in enumerate(tqdm(list(reader), desc=f"Rendering {scene_name}")):
            img_name = row["image_name"]
            qw, qx, qy, qz = float(row["qw"]), float(row["qx"]), float(row["qy"]), float(row["qz"])
            tx, ty, tz = float(row["tx"]), float(row["ty"]), float(row["tz"])
            fx, fy = float(row["fx"]), float(row["fy"])
            width, height = int(row["width"]), int(row["height"])
            cx = float(row["cx"]) if "cx" in row else width / 2.0
            cy = float(row["cy"]) if "cy" in row else height / 2.0
            
            # Convert COLMAP pose to R, T
            qvec = np.array([qw, qx, qy, qz])
            R = qvec2rotmat(qvec)
            T = np.array([tx, ty, tz])
            
            # The rotation matrix needs to be transposed for 3DGS Camera object
            R = np.transpose(R)
            
            # Use "Distort-Back" rendering if we have radial distortion parameters
            opencv_dist = np.array([
                radial_coeffs[0],  # k1
                radial_coeffs[1],  # k2
                0.0,  # p1
                0.0,  # p2
                0.0,  # k3
                0.0,  # k4
                0.0, 0.0
            ], dtype=np.float32)

            if np.any(opencv_dist) and distortion_mode == "legacy_remap":
                # Build K in OpenCV convention
                K = np.array([
                    [fx, 0.0, cx],
                    [0.0, fy, cy],
                    [0.0, 0.0, 1.0]
                ], dtype=np.float32)

                # Replicate undistortion pipeline exactly from vtairace_3D_BTS
                newK, roi = cv2.getOptimalNewCameraMatrix(K, opencv_dist, (width, height), 0)
                rx, ry, rw_roi, rh_roi = roi

                # Crop to ROI and adjust principal point
                newK_cropped = newK.copy()
                newK_cropped[0, 2] -= rx
                newK_cropped[1, 2] -= ry

                fx_u = float(newK_cropped[0, 0])
                fy_u = float(newK_cropped[1, 1])
                cx_u = float(newK_cropped[0, 2])
                cy_u = float(newK_cropped[1, 2])

                # Map every distorted pixel to its location in the cropped undistorted space
                y_grid, x_grid = np.mgrid[0:height, 0:width]
                pts_dist = np.stack([x_grid, y_grid], axis=-1).astype(np.float32).reshape(-1, 1, 2)
                pts_undist_norm = cv2.undistortPoints(pts_dist, K, opencv_dist, None, None)

                # Convert normalized coords to pixel coords in newK space, then shift for ROI crop
                pts_undist_pixel = pts_undist_norm.copy()
                pts_undist_pixel[:, 0, 0] = pts_undist_norm[:, 0, 0] * newK[0, 0] + newK[0, 2] - rx
                pts_undist_pixel[:, 0, 1] = pts_undist_norm[:, 0, 1] * newK[1, 1] + newK[1, 2] - ry

                map_x_raw = pts_undist_pixel[:, 0, 0].reshape(height, width)
                map_y_raw = pts_undist_pixel[:, 0, 1].reshape(height, width)

                # Dynamic padding to ensure no pixels map outside the rendered frame
                min_x = np.min(map_x_raw)
                max_x = np.max(map_x_raw)
                min_y = np.min(map_y_raw)
                max_y = np.max(map_y_raw)

                pad_left = max(0, int(np.ceil(-min_x)) + 2)
                pad_right = max(0, int(np.ceil(max_x - rw_roi)) + 2)
                pad_top = max(0, int(np.ceil(-min_y)) + 2)
                pad_bottom = max(0, int(np.ceil(max_y - rh_roi)) + 2)

                render_w = rw_roi + pad_left + pad_right
                render_h = rh_roi + pad_top + pad_bottom

                # Padded camera intrinsics for rendering
                fx_model = fx_u
                fy_model = fy_u
                cx_model = cx_u + pad_left
                cy_model = cy_u + pad_top

                FovY_u = focal2fov(fy_model, render_h)
                FovX_u = focal2fov(fx_model, render_w)

                dummy_image = Image.new("RGB", (render_w, render_h))

                cam = Camera(resolution=(render_w, render_h),
                             colmap_id=idx, 
                             R=R, T=T, 
                             FoVx=FovX_u, FoVy=FovY_u, 
                             depth_params=None,
                             image=dummy_image, 
                             invdepthmap=None,
                             image_name=img_name.split(".")[0], 
                             uid=idx,
                             data_device="cuda",
                             radial_coeffs=None)

                # Pass custom attributes to bypass FoV conversion in gaussian_renderer
                cam.fx = fx_model
                cam.fy = fy_model
                cam.cx = cx_model
                cam.cy = cy_model

                render_pkg = render(cam, gaussians, pipeline, background)
                image_tensor = render_pkg["render"]

                # Post-process: convert to numpy and remap back using Lanczos4 interpolation
                img_np = image_tensor.permute(1, 2, 0).detach().cpu().numpy()
                img_np = np.clip(img_np * 255, 0, 255).astype(np.uint8)

                map_x = (map_x_raw + pad_left).astype(np.float32)
                map_y = (map_y_raw + pad_top).astype(np.float32)
                
                rgb = cv2.remap(img_np, map_x, map_y, cv2.INTER_LANCZOS4)
                img = Image.fromarray(rgb)

            else:
                # Native gsplat projection.  Scaling K together with the output
                # resolution implements true supersampling without changing FoV.
                render_width = max(width, int(round(width * supersample)))
                render_height = max(height, int(round(height * supersample)))
                sx = render_width / width
                sy = render_height / height
                fx_render, fy_render = fx * sx, fy * sy
                cx_render, cy_render = cx * sx, cy * sy
                FovY = focal2fov(fy_render, render_height)
                FovX = focal2fov(fx_render, render_width)
                
                dummy_image = Image.new("RGB", (render_width, render_height))
                
                native_radial = radial_coeffs if distortion_mode == "native" and np.any(radial_coeffs) else None
                cam = Camera(resolution=(render_width, render_height),
                             colmap_id=idx, 
                             R=R, T=T, 
                             FoVx=FovX, FoVy=FovY, 
                             depth_params=None,
                             image=dummy_image, 
                             invdepthmap=None,
                             image_name=img_name.split(".")[0], 
                             uid=idx,
                             data_device="cuda",
                             radial_coeffs=native_radial)
                
                cam.fx = fx_render
                cam.fy = fy_render
                cam.cx = cx_render
                cam.cy = cy_render

                render_pkg = render(cam, gaussians, pipeline, background)
                image_tensor = render_pkg["render"]
                
                img_np = image_tensor.permute(1, 2, 0).detach().cpu().numpy()
                img_np = np.clip(np.rint(img_np * 255), 0, 255).astype(np.uint8)
                if (render_width, render_height) != (width, height):
                    interpolation = {
                        "area": cv2.INTER_AREA,
                        "lanczos": cv2.INTER_LANCZOS4,
                        "linear": cv2.INTER_LINEAR,
                    }.get(downsample_filter)
                    if interpolation is None:
                        raise ValueError(
                            f"Unknown downsample_filter={downsample_filter!r}; "
                            "expected area, lanczos, or linear"
                        )
                    img_np = cv2.resize(img_np, (width, height), interpolation=interpolation)
                img = Image.fromarray(img_np)
            
            # Use original extension from test_poses.csv if it exists
            # Otherwise use the requested format/extension
            ext_in_csv = os.path.splitext(img_name)[1]
            if ext_in_csv:
                out_img_name = img_name
                save_fmt = "JPEG" if ext_in_csv.lower() in [".jpg", ".jpeg"] else "PNG"
            else:
                out_img_name = img_name + ext
                save_fmt = "JPEG" if fmt == "jpeg" else "PNG"
            
            out_path = os.path.join(out_scene_dir, out_img_name)
            if save_fmt == "JPEG":
                img.save(out_path, "JPEG", quality=int(jpeg_quality), subsampling=0)
            else:
                img.save(out_path, "PNG")

def create_submission_zip(output_dir, zip_name="submission.zip"):
    import shutil
    logger.info(f"Zipping {output_dir} into {zip_name}...")
    shutil.make_archive(zip_name.replace(".zip", ""), 'zip', output_dir)
    logger.info("Done!")
