import os
import glob
import logging
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch

try:
    from transformers import pipeline
except ImportError:
    pipeline = None

logger = logging.getLogger(__name__)

def estimate_scene_depth(scene_dir, device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    Generate depth maps for all training images in a scene using Depth Anything V2.
    """
    if pipeline is None:
        logger.error("transformers library is not installed. Please install it to use depth estimation.")
        return False

    train_img_dir = os.path.join(scene_dir, "train", "images")
    depth_dir = os.path.join(scene_dir, "train", "depth")
    
    if not os.path.exists(train_img_dir):
        logger.warning(f"No train images found at {train_img_dir}")
        return False

    image_paths = sorted(glob.glob(os.path.join(train_img_dir, "*")))
    if not image_paths:
        return False

    os.makedirs(depth_dir, exist_ok=True)
    
    # We will gather images that still need depth mapping
    depth_params = {}
    to_process = []
    
    for img_path in image_paths:
        filename = os.path.basename(img_path)
        base_name = os.path.splitext(filename)[0]
        png_path = os.path.join(depth_dir, f"{base_name}.png")
        
        # Default parameter for each image
        depth_params[base_name] = {"scale": 1.0, "offset": 0.0}
        
        is_valid = False
        if os.path.exists(png_path) and os.path.getsize(png_path) > 0:
            import cv2
            test_img = cv2.imread(png_path, -1)
            if test_img is not None:
                is_valid = True
                
        if not is_valid:
            to_process.append((img_path, png_path))

    if not to_process:
        logger.info(f"Depth maps already exist for {scene_dir}. Skipping depth estimation.")
    else:
        logger.info(f"Loading DepthAnythingV2 model on {device}...")
        # Load pipeline
        pipe = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf", device=device)
        
        batch_size = 8
        logger.info(f"Estimating depth for {len(to_process)} remaining images in batches of {batch_size}...")
        
        for i in tqdm(range(0, len(to_process), batch_size), desc="Depth Estimation"):
            batch = to_process[i:i+batch_size]
            imgs = []
            valid_batch = []
            
            for img_path, png_path in batch:
                try:
                    imgs.append(Image.open(img_path).convert("RGB"))
                    valid_batch.append((img_path, png_path))
                except Exception as e:
                    logger.error(f"Failed to load {img_path}: {e}")
            
            if not imgs:
                continue
                
            try:
                # Perform batched inference
                results = pipe(imgs, batch_size=len(imgs))
                
                # If pipeline output is a single dict (when batch_size=1 and only 1 image passed)
                if not isinstance(results, list):
                    results = [results]
                    
                for (_, png_path), result in zip(valid_batch, results):
                    depth_image = result["depth"]
                    depth_image.save(png_path)
            except Exception as e:
                logger.error(f"Failed batch depth estimation at index {i}: {e}")

    # Write depth_params.json to train/sparse/0/ directory
    sparse_dir = os.path.join(scene_dir, "train", "sparse", "0")
    os.makedirs(sparse_dir, exist_ok=True)
    depth_params_file = os.path.join(sparse_dir, "depth_params.json")
    
    with open(depth_params_file, "w") as f:
        import json
        json.dump(depth_params, f, indent=4)
        
    logger.info(f"Saved {depth_params_file}")
    logger.info("Depth estimation complete.")
    return True

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    if len(sys.argv) > 1:
        estimate_scene_depth(sys.argv[1])
    else:
        print("Usage: python depth_estimator.py <path_to_scene_dir>")
