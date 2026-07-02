import os
import glob
import struct
import logging

logger = logging.getLogger(__name__)

def list_scenes(base_dir):
    """
    List all scenes in the VAI_NVS_DATA public_set or private_set.
    """
    scenes = []
    if not os.path.exists(base_dir):
        return scenes
    for d in os.listdir(base_dir):
        if os.path.isdir(os.path.join(base_dir, d)):
            scenes.append(d)
    return sorted(scenes)

def analyze_scene(scene_dir):
    """
    Analyze a specific scene directory.
    Returns statistics like number of train images, test images, etc.
    """
    train_img_dir = os.path.join(scene_dir, "train", "images")
    test_img_dir = os.path.join(scene_dir, "test", "images")
    sparse_dir = os.path.join(scene_dir, "train", "sparse", "0")
    test_csv = os.path.join(scene_dir, "test", "test_poses.csv")
    
    num_train_images = len(glob.glob(os.path.join(train_img_dir, "*"))) if os.path.exists(train_img_dir) else 0
    num_test_images = len(glob.glob(os.path.join(test_img_dir, "*"))) if os.path.exists(test_img_dir) else 0
    
    has_sparse = os.path.exists(sparse_dir) and os.path.exists(os.path.join(sparse_dir, "points3D.bin"))
    
    # Read points3D.bin to get number of points
    num_points = 0
    if has_sparse:
        try:
            num_points = read_points3d_count(os.path.join(sparse_dir, "points3D.bin"))
        except Exception as e:
            logger.error(f"Could not read points3D.bin: {e}")
            
    num_test_poses = 0
    if os.path.exists(test_csv):
        with open(test_csv, "r") as f:
            lines = f.readlines()
            if len(lines) > 1:
                num_test_poses = len(lines) - 1 # exclude header
                
    return {
        "num_train_images": num_train_images,
        "num_test_images": num_test_images,
        "has_sparse_reconstruction": has_sparse,
        "num_sparse_points": num_points,
        "num_test_poses": num_test_poses
    }

def read_points3d_count(path):
    """
    Reads the number of 3D points from COLMAP points3D.bin file.
    """
    with open(path, "rb") as fid:
        num_points = struct.unpack("<Q", fid.read(8))[0]
    return num_points
