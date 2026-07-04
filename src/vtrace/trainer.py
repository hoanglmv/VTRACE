import os
import subprocess
import sys
import logging
from .quality_filter import QualityFilter

logger = logging.getLogger(__name__)

def train_scene(scene_dir, output_dir, iterations=30000, resolution=1, data_device="cpu", sh_degree=2, gs_path="src/vtrace/gaussian-splatting", early_stopping_start_iter=7000, early_stopping_window_iters=2000, early_stopping_rel_change=0.005, lambda_opacity=0.0, lambda_scale=0.0, lambda_dssim=0.2, lambda_edge=0.0):
    """
    Trains the 3DGS model for a given scene.
    scene_dir: path to the scene directory (e.g. VAI_NVS_DATA/phase1/public_set/HCM0181)
    output_dir: where to save the trained model
    """
    train_script = os.path.join(gs_path, "train.py")
    if not os.path.exists(train_script):
        raise FileNotFoundError(f"train.py not found in {gs_path}. Have you run setup.py?")
    
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
        "--disable_viewer"
    ]
    
    image_dir = os.path.join(source_path, "images")
    q_filter = QualityFilter(image_dir)
    
    logger.info(f"Running training command for {scene_dir}...")
    
    try:
        q_filter.apply()
        # Popen can be used for streaming output or subprocess.run for blocking
        subprocess.run(cmd, check=True)
        logger.info(f"Training completed for scene {scene_dir}. Model saved to {output_dir}")
    except subprocess.CalledProcessError as e:
        logger.error(f"Error during training scene {scene_dir}: {e}")
    finally:
        q_filter.restore()
