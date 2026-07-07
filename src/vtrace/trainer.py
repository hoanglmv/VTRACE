import os
import subprocess
import sys
import logging
from .quality_filter import QualityFilter

logger = logging.getLogger(__name__)

def train_scene(scene_dir, output_dir, iterations=30000, resolution=1, data_device="cpu", sh_degree=2, gs_path="src/vtrace/gaussian-splatting", early_stopping_start_iter=7000, early_stopping_window_iters=5000, early_stopping_rel_change=0.00001, lambda_opacity=0.0, lambda_scale=0.0, lambda_dssim=0.2, lambda_edge=0.0):
    """
    Trains the 3DGS model for a given scene.
    scene_dir: path to the scene directory (e.g. VAI_NVS_DATA/phase1/public_set/HCM0181)
    output_dir: where to save the trained model
    """
    # Resolve gs_path robustly if not found directly
    train_script = os.path.join(gs_path, "train.py")
    if not os.path.exists(train_script):
        # Try resolving relative to this trainer.py file
        alternative_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "gaussian-splatting"))
        alternative_script = os.path.join(alternative_path, "train.py")
        if os.path.exists(alternative_script):
            gs_path = alternative_path
            train_script = alternative_script
        else:
            raise FileNotFoundError(f"train.py not found in {gs_path} or fallback {alternative_path}. Have you run setup.py?")
            
    # Automatically apply patches to the cloned gaussian-splatting folder before training
    import shutil
    patches = {
        "dataset_readers.py": "scene/dataset_readers.py",
        "train.py": "train.py",
        "arguments_init.py": "arguments/__init__.py",
        "cameras.py": "scene/cameras.py",
        "gaussian_model.py": "scene/gaussian_model.py",
        "camera_utils.py": "utils/camera_utils.py"
    }
    patches_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "patches"))
    for patch_name, relative_dest in patches.items():
        src_file = os.path.join(patches_dir, patch_name)
        dest_file = os.path.join(gs_path, relative_dest)
        if os.path.exists(src_file):
            try:
                os.makedirs(os.path.dirname(dest_file), exist_ok=True)
                shutil.copy2(src_file, dest_file)
                logger.info(f"Auto-patched: {patch_name} -> {relative_dest}")
            except Exception as e:
                logger.warning(f"Failed to auto-patch {patch_name}: {e}")
    
    # The source data is in scene_dir/train
    source_path = os.path.join(scene_dir, "train")
    if not os.path.exists(source_path):
        raise FileNotFoundError(f"Source path {source_path} does not exist.")
        
    python_exe = sys.executable
    
    # Standard 3DGS training command
    cmd = [
        python_exe, train_script,
        "-s", source_path,
        "-m", output_dir,
        "--iterations", str(iterations),
        "-r", str(resolution), # resolution factor
        "--data_device", data_device,
        "--sh_degree", str(sh_degree),
        "--densify_grad_threshold", "0.0006",
        "--early_stopping_start_iter", str(early_stopping_start_iter),
        "--early_stopping_window_iters", str(early_stopping_window_iters),
        "--early_stopping_rel_change", str(early_stopping_rel_change),
        "--lambda_opacity", str(lambda_opacity),
        "--lambda_scale", str(lambda_scale),
        "--lambda_dssim", str(lambda_dssim),
        "--lambda_edge", str(lambda_edge),
        "--depths", "depth",
        "--disable_viewer"
    ]
    
    image_dir = os.path.join(source_path, "images")
    q_filter = QualityFilter(image_dir)
    
    scene_name = os.path.basename(scene_dir.rstrip("/\\"))
    log_dir = os.path.abspath(os.path.join(output_dir, "..", "..", "logs"))
    os.makedirs(log_dir, exist_ok=True)
    train_log_path = os.path.join(log_dir, f"{scene_name}_train.log")
    
    logger.info(f"Running training command for {scene_name}...")
    logger.info(f"Training output will be saved to: {train_log_path}")
    
    try:
        q_filter.apply()
        with open(train_log_path, "w", encoding="utf-8") as log_file:
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
            while True:
                char = process.stdout.read(1)
                if not char:
                    break
                sys.stdout.write(char)
                sys.stdout.flush()
                log_file.write(char)
            process.wait()
            if process.returncode != 0:
                raise subprocess.CalledProcessError(process.returncode, cmd)
        logger.info(f"Training completed for scene {scene_dir}. Model saved to {output_dir}")
    except subprocess.CalledProcessError as e:
        logger.error(f"Error during training scene {scene_dir}: {e}")
    finally:
        q_filter.restore()
