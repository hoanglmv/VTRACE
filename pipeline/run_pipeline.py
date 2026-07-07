import os
import argparse
import logging
import sys
import yaml

# Ensure src module is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.vtrace.data_utils import list_scenes, analyze_scene
from src.vtrace.trainer import train_scene
from src.vtrace.renderer import render_scene, create_submission_zip
from src.vtrace.depth_estimator import estimate_scene_depth
from src.vtrace.post_processor import post_process_scene_renders

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
            "format": "png"
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
    
    # New regularization parameters
    lambda_opacity = config_data["training"].get("lambda_opacity", 0.0)
    lambda_scale = config_data["training"].get("lambda_scale", 0.0)
    lambda_dssim = config_data["training"].get("lambda_dssim", 0.2)
    lambda_edge = config_data["training"].get("lambda_edge", 0.0)

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
        
        if not stats["has_sparse_reconstruction"]:
            logger.warning(f"Scene {scene} missing sparse reconstruction. Skipping.")
            continue
            
        scene_model_dir = os.path.join(models_dir, scene)
        
        if not skip_training:
            logger.info(f"--- Estimating Depth for {scene} ---")
            estimate_scene_depth(scene_dir, device=data_device)
            
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
                lambda_edge=lambda_edge
            )
        
        logger.info(f"--- Rendering {scene} ---")
        render_scene(scene, scene_dir, scene_model_dir, submission_dir, render_format=render_format)
        
        logger.info(f"--- Post-processing {scene} ---")
        post_process_scene_renders(submission_dir, scene)
        
    logger.info("--- Creating Submission Archive ---")
    create_submission_zip(submission_dir, os.path.join(out_dir, "submission_round1.zip"))

if __name__ == "__main__":
    main()
