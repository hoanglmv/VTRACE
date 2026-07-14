import os
import argparse
import logging
import sys
import yaml
import shutil

# Ensure src module is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Automatically apply patches to the cloned gaussian-splatting folder BEFORE importing any local modules
# to prevent Python from importing unpatched cached modules
def apply_pre_imports_patches():
    gs_path = "src/vtrace/gaussian-splatting"
    # Resolve relative to run_pipeline.py file if not found
    if not os.path.exists(os.path.join(gs_path, "train.py")):
        gs_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src", "vtrace", "gaussian-splatting"))
        
    patches = {
        "dataset_readers.py": "scene/dataset_readers.py",
        "train.py": "train.py",
        "arguments_init.py": "arguments/__init__.py",
        "cameras.py": "scene/cameras.py",
        "gaussian_model.py": "scene/gaussian_model.py",
        "camera_utils.py": "utils/camera_utils.py",
        "gaussian_renderer.py": "gaussian_renderer/__init__.py"
    }
    patches_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src", "vtrace", "patches"))
    for patch_name, relative_dest in patches.items():
        src_file = os.path.join(patches_dir, patch_name)
        dest_file = os.path.join(gs_path, relative_dest)
        if os.path.exists(src_file):
            try:
                os.makedirs(os.path.dirname(dest_file), exist_ok=True)
                shutil.copy2(src_file, dest_file)
            except Exception as e:
                print(f"Failed to pre-patch {patch_name}: {e}")

apply_pre_imports_patches()

from src.vtrace.data_utils import list_scenes, analyze_scene
from src.vtrace.trainer import train_scene
from src.vtrace.renderer import render_scene, create_submission_zip
from src.vtrace.depth_estimator import estimate_scene_depth
from src.vtrace.post_processor import post_process_scene_renders
from src.vtrace.densify import densify_scene_point_cloud

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="VTRACE 3DGS Pipeline")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML configuration file")
    parser.add_argument("--data-dir", type=str, default=None, help="Path to VAI_NVS_DATA directory")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory")
    parser.add_argument("--iterations", type=int, default=None, help="Number of training iterations")
    parser.add_argument("--resolution", type=int, default=None, help="Resolution scaling factor")
    parser.add_argument("--data-device", type=str, default=None, choices=["cuda", "cpu"], help="Device to store data (cuda or cpu)")
    parser.add_argument("--sh-degree", type=int, default=None, help="Spherical Harmonics degree")
    parser.add_argument("--render-format", type=str, default=None, choices=["png", "jpg", "jpeg"], help="Output format for rendered images (png, jpg, jpeg)")
    parser.add_argument("--skip-training", action="store_true", default=None, help="Skip training and only render")
    args = parser.parse_args()

    # Default configuration values
    config_data = {
        "pipeline": {
            "data_dir": "./VAI_NVS_DATA/phase1/public_set",
            "output_dir": "./output",
        },
        "training": {
            "iterations": 30000,
            "resolution": 1,
            "data_device": "cpu",
            "sh_degree": 2,
        },
        "render": {
            "skip_training": False,
            "format": "png",
            "antialiasing": False,
            "distortion_mode": "native",
            "with_ut": True,
            "with_eval3d": False,
            "supersample": 1.0,
            "downsample_filter": "area",
            "jpeg_quality": 100,
            "checkpoint_iteration": "latest",
            "post_process": False,
        }
    }

    # Load YAML if provided
    if args.config:
        logger.info(f"Loading configuration from {args.config}")
        with open(args.config, "r") as f:
            yaml_data = yaml.safe_load(f)
            if yaml_data:
                if "pipeline" in yaml_data and yaml_data["pipeline"]:
                    config_data["pipeline"].update(yaml_data["pipeline"])
                if "training" in yaml_data and yaml_data["training"]:
                    config_data["training"].update(yaml_data["training"])
                if "render" in yaml_data and yaml_data["render"]:
                    config_data["render"].update(yaml_data["render"])

    # Merge configuration with command-line overrides
    data_dir = os.path.abspath(args.data_dir if args.data_dir is not None else config_data["pipeline"]["data_dir"])
    out_dir = os.path.abspath(args.output_dir if args.output_dir is not None else config_data["pipeline"]["output_dir"])
    iterations = args.iterations if args.iterations is not None else config_data["training"]["iterations"]
    resolution = args.resolution if args.resolution is not None else config_data["training"]["resolution"]
    data_device = args.data_device if args.data_device is not None else config_data["training"].get("data_device", "cpu")
    sh_degree = args.sh_degree if args.sh_degree is not None else config_data["training"].get("sh_degree", 2)
    skip_training = args.skip_training if args.skip_training is not None else config_data["render"]["skip_training"]
    render_format = args.render_format if args.render_format is not None else config_data["render"].get("format", "png")
    render_antialiasing = config_data["render"].get(
        "antialiasing", config_data["training"].get("antialiasing", False)
    )
    distortion_mode = config_data["render"].get("distortion_mode", "native")
    render_with_ut = config_data["render"].get("with_ut", True)
    render_with_eval3d = config_data["render"].get("with_eval3d", False)
    supersample = float(config_data["render"].get("supersample", 1.0))
    downsample_filter = config_data["render"].get("downsample_filter", "area")
    jpeg_quality = int(config_data["render"].get("jpeg_quality", 100))
    checkpoint_iteration = config_data["render"].get("checkpoint_iteration", "latest")
    post_process = bool(config_data["render"].get("post_process", False))
    appearance_mode = config_data["render"].get("appearance_mode", "none")
    appearance_neighbors = int(config_data["render"].get("appearance_neighbors", 8))
    appearance_temperature = float(config_data["render"].get("appearance_temperature", 0.35))
    
    # New regularization parameters
    lambda_opacity = config_data["training"].get("lambda_opacity", 0.0)
    lambda_scale = config_data["training"].get("lambda_scale", 0.0)
    lambda_dssim = config_data["training"].get("lambda_dssim", 0.2)
    lambda_edge = config_data["training"].get("lambda_edge", 0.0)
    densify_until_iter = config_data["training"].get("densify_until_iter", 15000)
    antialiasing = config_data["training"].get("antialiasing", False)
    estimate_depth = config_data["training"].get("estimate_depth", True)
    densify_point_cloud = config_data["training"].get("densify_point_cloud", True)
    densification_strategy = config_data["training"].get("densification_strategy", "mcmc")
    absgrad = config_data["training"].get("absgrad", False)
    revised_opacity = config_data["training"].get("revised_opacity", False)
    grow_grad2d = config_data["training"].get("grow_grad2d", 0.0002)
    mcmc_cap_max = config_data["training"].get("mcmc_cap_max", 3_000_000)
    mcmc_noise_lr = config_data["training"].get("mcmc_noise_lr", 500_000.0)
    lambda_frequency = config_data["training"].get("lambda_frequency", 0.0)
    frequency_start_iter = config_data["training"].get("frequency_start_iter", 3000)
    frequency_ramp_iters = config_data["training"].get("frequency_ramp_iters", 10000)
    frequency_max_resolution = config_data["training"].get("frequency_max_resolution", 512)
    depth_l1_weight_init = config_data["training"].get("depth_l1_weight_init", 1.0)
    depth_l1_weight_final = config_data["training"].get("depth_l1_weight_final", 0.01)
    optimize_exposure = config_data["training"].get("optimize_exposure", False)
    lambda_exposure = config_data["training"].get("lambda_exposure", 0.001)

    models_dir = os.path.join(out_dir, "models")
    submission_dir = os.path.join(out_dir, "submission")
    
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(submission_dir, exist_ok=True)
    
    # Set up pipeline log file
    log_dir = os.path.join(out_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    pipeline_log_path = os.path.join(log_dir, "pipeline.log")
    
    file_handler = logging.FileHandler(pipeline_log_path, encoding='utf-8')
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logging.getLogger().addHandler(file_handler)
    logger.info(f"Pipeline log will be saved to {pipeline_log_path}")
    
    scenes = list_scenes(data_dir)
    logger.info(f"Found {len(scenes)} scenes in {data_dir}")
    
    for scene in scenes:
        scene_dir = os.path.join(data_dir, scene)
        stats = analyze_scene(scene_dir)
        logger.info(f"Scene {scene} stats: {stats}")
        
        scene_model_dir = os.path.join(models_dir, scene)
        
        do_train = not skip_training
        if not stats["has_sparse_reconstruction"]:
            logger.warning(f"Scene {scene} missing sparse reconstruction. Skipping training, will generate dummy renders.")
            do_train = False
            
        if do_train:
            if estimate_depth:
                logger.info(f"--- Estimating Depth for {scene} ---")
                estimate_scene_depth(scene_dir, device=data_device)
            
            if densify_point_cloud:
                logger.info(f"--- Densifying Point Cloud for {scene} ---")
                densify_scene_point_cloud(scene_dir)
            
            logger.info(f"--- Training {scene} ---")
            train_scene(
                scene_dir, 
                scene_model_dir, 
                iterations=iterations, 
                resolution=resolution,
                data_device=data_device,
                sh_degree=sh_degree,
                lambda_opacity=lambda_opacity,
                lambda_scale=lambda_scale,
                lambda_dssim=lambda_dssim,
                lambda_edge=lambda_edge,
                densify_until_iter=densify_until_iter,
                antialiasing=antialiasing,
                densification_strategy=densification_strategy,
                absgrad=absgrad,
                revised_opacity=revised_opacity,
                grow_grad2d=grow_grad2d,
                mcmc_cap_max=mcmc_cap_max,
                mcmc_noise_lr=mcmc_noise_lr,
                lambda_frequency=lambda_frequency,
                frequency_start_iter=frequency_start_iter,
                frequency_ramp_iters=frequency_ramp_iters,
                frequency_max_resolution=frequency_max_resolution,
                depth_l1_weight_init=depth_l1_weight_init,
                depth_l1_weight_final=depth_l1_weight_final,
                optimize_exposure=optimize_exposure,
                lambda_exposure=lambda_exposure,
            )
        
        logger.info(f"--- Rendering {scene} ---")
        render_scene(
            scene,
            scene_dir,
            scene_model_dir,
            submission_dir,
            render_format=render_format,
            antialiasing=render_antialiasing,
            distortion_mode=distortion_mode,
            with_ut=render_with_ut,
            with_eval3d=render_with_eval3d,
            supersample=supersample,
            downsample_filter=downsample_filter,
            jpeg_quality=jpeg_quality,
            checkpoint_iteration=checkpoint_iteration,
            appearance_mode=appearance_mode,
            appearance_neighbors=appearance_neighbors,
            appearance_temperature=appearance_temperature,
        )
        
        if post_process:
            logger.info(f"--- Post-processing {scene} ---")
            post_process_scene_renders(submission_dir, scene)
        
    logger.info("--- Creating Submission Archive ---")
    create_submission_zip(submission_dir, os.path.join(out_dir, "submission_round1.zip"))

if __name__ == "__main__":
    main()
