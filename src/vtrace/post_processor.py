import os
import glob
import cv2
import numpy as np
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)

def enhance_image(image):
    """
    Apply Unsharp Masking and slight contrast enhancement
    to improve sharpness (SSIM/LPIPS) of 3DGS rendered images.
    """
    # 1. Unsharp Masking for Edge Enhancement
    # This sharpens the slightly soft edges common in vanilla 3DGS
    gaussian = cv2.GaussianBlur(image, (0, 0), 2.0)
    unsharp_image = cv2.addWeighted(image, 1.5, gaussian, -0.5, 0)
    
    # 2. Slight Contrast/Brightness Adjustment
    # Compensates for SH degree limitations in dark areas
    enhanced = cv2.convertScaleAbs(unsharp_image, alpha=1.02, beta=2)
    return enhanced

def post_process_scene_renders(submission_dir, scene):
    """
    Process all rendered images for a specific scene in the submission folder.
    """
    scene_dir = os.path.join(submission_dir, scene)
    if not os.path.exists(scene_dir):
        logger.warning(f"No renders found for scene {scene} at {scene_dir}")
        return False
        
    image_paths = glob.glob(os.path.join(scene_dir, "*.*"))
    # Filter for standard image formats
    image_paths = [p for p in image_paths if p.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not image_paths:
        return False
        
    logger.info(f"Post-processing {len(image_paths)} rendered images for {scene}...")
    
    for img_path in tqdm(image_paths, desc=f"Enhancing Renders ({scene})"):
        try:
            img = cv2.imread(img_path)
            if img is not None:
                enhanced = enhance_image(img)
                # Overwrite the rendered image with the enhanced version
                cv2.imwrite(img_path, enhanced)
        except Exception as e:
            logger.error(f"Failed to process {img_path}: {e}")
            
    return True
