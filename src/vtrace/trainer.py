import os
import subprocess
import sys
import logging

logger = logging.getLogger(__name__)

def train_scene(scene_dir, output_dir, iterations=30000, resolution=1, data_device="cpu", sh_degree=2, gs_path="src/vtrace/gaussian-splatting"):
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
        "--densify_grad_threshold", "0.0006"
    ]
    
    logger.info(f"Running training command for {scene_dir}...")
    
    try:
        # Popen can be used for streaming output or subprocess.run for blocking
        subprocess.run(cmd, check=True)
        logger.info(f"Training completed for scene {scene_dir}. Model saved to {output_dir}")
    except subprocess.CalledProcessError as e:
        logger.error(f"Error during training scene {scene_dir}: {e}")
