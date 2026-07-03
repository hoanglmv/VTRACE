import sys
import os
import csv
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "gaussian-splatting")))

try:
    from scene.gaussian_model import GaussianModel
    from gaussian_renderer import render
    from utils.graphics_utils import fov2focal, focal2fov
    from scene.cameras import Camera
except ImportError:
    logger.warning("3DGS modules not found. Ensure setup.py has been run and gaussian-splatting is cloned.")

class PipelineParams:
    def __init__(self):
        self.compute_cov3D_python = False
        self.convert_SHs_python = False
        self.debug = False
        self.antialiasing = False

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

def render_scene(scene_name, scene_dir, model_path, output_dir, render_format="png"):
    test_csv = os.path.join(scene_dir, "test", "test_poses.csv")
    if not os.path.exists(test_csv):
        logger.warning(f"No test_poses.csv found for {scene_name}")
        return

    # Determine SH degree from PLY file dynamically
    import math
    from plyfile import PlyData
    
    iter_dir = os.path.join(model_path, "point_cloud")
    if not os.path.exists(iter_dir):
        logger.warning(f"Model not found at {iter_dir}")
        return
        
    iterations = sorted([int(i.split("_")[-1]) for i in os.listdir(iter_dir) if "iteration_" in i])
    if not iterations:
        logger.warning("No iterations found")
        return
        
    latest_iter = iterations[-1]
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
    pipeline = PipelineParams()
    
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
            
            # Convert COLMAP pose to R, T
            qvec = np.array([qw, qx, qy, qz])
            R = qvec2rotmat(qvec)
            T = np.array([tx, ty, tz])
            
            # The rotation matrix needs to be transposed for 3DGS Camera object
            R = np.transpose(R)
            
            FovY = focal2fov(fy, height)
            FovX = focal2fov(fx, width)
            
            # Dummy PIL Image since Camera constructor converts it to Torch
            dummy_image = Image.new("RGB", (width, height))
            
            cam = Camera(resolution=(width, height),
                         colmap_id=idx, 
                         R=R, T=T, 
                         FoVx=FovX, FoVy=FovY, 
                         depth_params=None,
                         image=dummy_image, 
                         invdepthmap=None,
                         image_name=img_name.split(".")[0], 
                         uid=idx,
                         data_device="cuda")
            
            # Render
            render_pkg = render(cam, gaussians, pipeline, background)
            image_tensor = render_pkg["render"]
            
            # Save
            img_np = image_tensor.permute(1, 2, 0).detach().cpu().numpy()
            img_np = np.clip(img_np * 255, 0, 255).astype(np.uint8)
            img = Image.fromarray(img_np)
            
            # Use requested format/extension
            base_img_name = os.path.splitext(img_name)[0]
            out_img_name = base_img_name + ext
            
            out_path = os.path.join(out_scene_dir, out_img_name)
            if fmt == "jpeg":
                img.save(out_path, "JPEG", quality=90)
            else:
                img.save(out_path, "PNG")

def create_submission_zip(output_dir, zip_name="submission.zip"):
    import shutil
    logger.info(f"Zipping {output_dir} into {zip_name}...")
    shutil.make_archive(zip_name.replace(".zip", ""), 'zip', output_dir)
    logger.info("Done!")
