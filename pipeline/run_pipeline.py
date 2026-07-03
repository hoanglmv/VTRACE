import os
import argparse
import logging
import sys

# Ensure src module is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.vtrace.data_utils import list_scenes, analyze_scene
from src.vtrace.trainer import train_scene
from src.vtrace.renderer import render_scene, create_submission_zip

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="VTRACE 3DGS Pipeline")
    parser.add_argument("--data-dir", type=str, required=True, help="Path to VAI_NVS_DATA directory (e.g., ./VAI_NVS_DATA/phase1/public_set)")
    parser.add_argument("--output-dir", type=str, default="./output", help="Output directory for trained models and renders")
    parser.add_argument("--iterations", type=int, default=30000, help="Number of training iterations")
    parser.add_argument("--resolution", type=int, default=1, help="Resolution scaling factor")
    parser.add_argument("--skip-training", action="store_true", help="Skip training and only render")
    args = parser.parse_args()

    data_dir = os.path.abspath(args.data_dir)
    out_dir = os.path.abspath(args.output_dir)
    models_dir = os.path.join(out_dir, "models")
    submission_dir = os.path.join(out_dir, "submission")
    
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(submission_dir, exist_ok=True)
    
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
        
        if not args.skip_training:
            logger.info(f"--- Training {scene} ---")
            train_scene(scene_dir, scene_model_dir, iterations=args.iterations, resolution=args.resolution)
        
        logger.info(f"--- Rendering {scene} ---")
        render_scene(scene, scene_dir, scene_model_dir, submission_dir)
        
    logger.info("--- Creating Submission Archive ---")
    create_submission_zip(submission_dir, os.path.join(out_dir, "submission.zip"))

if __name__ == "__main__":
    main()
