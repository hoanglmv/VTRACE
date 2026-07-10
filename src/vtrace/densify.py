import os
import sys
import shutil
import struct
import logging
import numpy as np
from PIL import Image
from plyfile import PlyData, PlyElement

logger = logging.getLogger(__name__)

# Make sure gaussian-splatting packages are in path
gs_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "gaussian-splatting"))
if gs_path not in sys.path:
    sys.path.append(gs_path)

from scene.colmap_loader import read_extrinsics_binary, read_intrinsics_binary, CameraModel, qvec2rotmat

def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)

def read_points3D_binary_dict(path_to_model_file):
    """
    Custom fast binary parser for COLMAP points3D.bin that preserves 3D point IDs.
    """
    points3D = {}
    with open(path_to_model_file, "rb") as fid:
        num_points = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_points):
            binary_point_line_properties = read_next_bytes(
                fid, num_bytes=43, format_char_sequence="QdddBBBd")
            point3D_id = binary_point_line_properties[0]
            xyz = np.array(binary_point_line_properties[1:4])
            rgb = np.array(binary_point_line_properties[4:7])
            
            track_length = read_next_bytes(
                fid, num_bytes=8, format_char_sequence="Q")[0]
            fid.seek(8 * track_length, 1) # Skip track elements (each is 2 int32 = 8 bytes)
            
            points3D[point3D_id] = (xyz, rgb)
    return points3D

def storePly(path, xyz, rgb):
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    normals = np.zeros_like(xyz)
    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def voxel_down_sample_numpy(xyz, rgb, voxel_size):
    """
    Downsamples the point cloud using a voxel grid filter in pure numpy/scipy.
    For each voxel, we keep the first point/color.
    """
    if voxel_size is None or voxel_size <= 0:
        return xyz, rgb

    # Quantize coordinates to voxel grid
    coords = np.floor(xyz / voxel_size).astype(np.int32)
    
    # Sort by voxel coordinates to group identical voxels together
    idx = np.lexsort((coords[:, 2], coords[:, 1], coords[:, 0]))
    coords_sorted = coords[idx]
    xyz_sorted = xyz[idx]
    rgb_sorted = rgb[idx]
    
    # Find unique voxels
    mask = np.empty(len(coords_sorted), dtype=bool)
    mask[0] = True
    mask[1:] = np.any(coords_sorted[1:] != coords_sorted[:-1], axis=1)
    
    return xyz_sorted[mask], rgb_sorted[mask]

def densify_scene_point_cloud(scene_dir, subsample=6, max_depth_factor=3.0, min_anchors=6, voxel_size=0.05, force=False):
    """
    Densify the sparse point cloud of a scene using already estimated depth maps.
    """
    sparse_dir = os.path.join(scene_dir, "train", "sparse", "0")
    depth_dir = os.path.join(scene_dir, "train", "depth")
    image_dir = os.path.join(scene_dir, "train", "images")
    
    ply_path = os.path.join(sparse_dir, "points3D.ply")
    ply_backup = os.path.join(sparse_dir, "points3D_sparse.ply")
    
    if os.path.exists(ply_backup) and not force:
        logger.info(f"Scene in {scene_dir} is already densified. Skipping.")
        return True

    # Check if sparse point cloud exists as binary, text, or PLY
    bin_path = os.path.join(sparse_dir, "points3D.bin")
    txt_path = os.path.join(sparse_dir, "points3D.txt")
    
    if not os.path.exists(bin_path) and not os.path.exists(txt_path) and not os.path.exists(ply_path):
        logger.warning(f"No points3D file found in {sparse_dir}, skipping densification.")
        return False
        
    # Read original sparse points first to get coordinates and colors
    try:
        if os.path.exists(bin_path):
            points3D_dict = read_points3D_binary_dict(bin_path)
            existing_xyz = np.array([pt[0] for pt in points3D_dict.values()], dtype=np.float32)
            existing_rgb = np.array([pt[1] for pt in points3D_dict.values()], dtype=np.uint8)
        else:
            # Fallback to load from PLY if bin not available
            if not os.path.exists(ply_path):
                logger.warning(f"No points3D.bin or PLY found in {sparse_dir}, skipping.")
                return False
            plydata = PlyData.read(ply_path)
            vertices = plydata['vertex']
            existing_xyz = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
            existing_rgb = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T
            # Reconstruct dictionary from PLY for anchors search
            points3D_dict = {i: (existing_xyz[i], existing_rgb[i]) for i in range(len(existing_xyz))}
    except Exception as e:
        logger.error(f"Failed to load sparse point cloud coordinates: {e}")
        return False
        
    if len(existing_xyz) == 0:
        logger.warning(f"Sparse point cloud is empty for {scene_dir}, skipping densification.")
        return False

    # Backup the original PLY file
    if os.path.exists(ply_path):
        shutil.copy2(ply_path, ply_backup)
        logger.info(f"Backed up sparse point cloud to {ply_backup}")
    else:
        # Create a temporary sparse PLY as backup
        storePly(ply_backup, existing_xyz, existing_rgb)
        logger.info(f"Created sparse point cloud backup at {ply_backup}")
    
    # Load COLMAP extrinsic and intrinsic cameras
    cameras_extrinsic_file = os.path.join(sparse_dir, "images.bin")
    cameras_intrinsic_file = os.path.join(sparse_dir, "cameras.bin")
    
    if not os.path.exists(cameras_extrinsic_file) or not os.path.exists(cameras_intrinsic_file):
        logger.error("COLMAP camera binary files not found, cannot densify.")
        return False
        
    try:
        images_data = read_extrinsics_binary(cameras_extrinsic_file)
        cameras_data = read_intrinsics_binary(cameras_intrinsic_file)
    except Exception as e:
        logger.error(f"Failed to read COLMAP camera files: {e}")
        return False
        
    scene_center = existing_xyz.mean(axis=0)
    scene_radius = float(np.linalg.norm(existing_xyz - scene_center, axis=1).max())
    max_dist = max_depth_factor * scene_radius
    logger.info(f"Scene center: {scene_center}, radius: {scene_radius:.2f}m, max allowed distance: {max_dist:.2f}m")
    
    all_new_xyz = []
    all_new_rgb = []
    n_processed = 0
    
    for img_id, img in images_data.items():
        base_name = os.path.splitext(img.name)[0]
        png_path = os.path.join(depth_dir, f"{base_name}.png")
        img_path = os.path.join(image_dir, img.name)
        
        if not os.path.exists(png_path) or not os.path.exists(img_path):
            continue
            
        cam = cameras_data[img.camera_id]
        params = list(cam.params)
        
        # SIMPLE_RADIAL: (f, cx, cy, k1); OPENCV: (fx, fy, cx, cy, ...)
        if len(params) >= 4 and params[0] != params[1]:
            # OPENCV
            fx, fy = float(params[0]), float(params[1])
            cx, cy = float(params[2]), float(params[3])
        else:
            # SIMPLE_RADIAL / SIMPLE_PINHOLE
            fx = fy = float(params[0])
            cx = float(params[1])
            cy = float(params[2])
            
        # COLMAP Pose (W2C)
        R_w2c = qvec2rotmat(img.qvec)
        t_w2c = np.array(img.tvec)
        R_c2w = R_w2c.T
        t_c2w = -R_c2w @ t_w2c
        
        # Load image and depth
        try:
            img_np = np.array(Image.open(img_path).convert("RGB"))
            H, W = img_np.shape[:2]
            
            depth_img = np.array(Image.open(png_path))
            # Depth map is saved as relative inverse depth (0-255).
            # Convert to relative depth (z = 1.0 / (depth_inv + 1e-5))
            depth_norm = depth_img.astype(np.float32) / (65535.0 if depth_img.dtype == np.uint16 else 255.0)
            depth_rel = 1.0 / (depth_norm + 1e-5)
        except Exception as e:
            logger.error(f"Failed to read image/depth for {img.name}: {e}")
            continue
        
        # Map COLMAP sparse points to image plane to act as anchor points
        anc_metric = []
        anc_pred = []
        
        for p2d, pt3d_id in zip(img.xys, img.point3D_ids):
            pt3d_id = int(pt3d_id)
            if pt3d_id == -1 or pt3d_id not in points3D_dict:
                continue

            u, v = float(p2d[0]), float(p2d[1])
            ui, vi = int(round(u)), int(round(v))
            if not (0 <= ui < W and 0 <= vi < H):
                continue
                
            xyz = points3D_dict[pt3d_id][0]
            # Convert world to camera space
            p_cam = R_w2c @ xyz + t_w2c
            z_metric = float(p_cam[2])
            if z_metric < 0.5:
                continue
                
            anc_metric.append(z_metric)
            anc_pred.append(float(depth_rel[vi, ui]))
            
        if len(anc_metric) < min_anchors:
            logger.warning(f"Image {img.name} only has {len(anc_metric)} anchors (< {min_anchors}), skipping.")
            continue
            
        # Least-squares fit: z_metric = scale * z_pred + shift
        A = np.column_stack([anc_pred, np.ones(len(anc_pred))])
        b = np.array(anc_metric)
        (scale, shift), *_ = np.linalg.lstsq(A, b, rcond=None)
        scale, shift = float(scale), float(shift)
        
        # Filter outliers based on reconstruction error
        pred_check = scale * np.array(anc_pred) + shift
        rmse = float(np.sqrt(np.mean((pred_check - b) ** 2)))
        median_depth = float(np.median(b))
        
        if rmse > 0.5 * median_depth:
            logger.warning(f"Image {img.name} has high depth RMSE {rmse:.2f}m (> 50% median {median_depth:.2f}m), skipping.")
            continue
            
        # Linear scale predicted depth map to metric space
        depth_metric = (scale * depth_rel + shift).astype(np.float32)
        
        # Filter out depth edges/boundaries (which have unreliable depth predictions)
        # Compute simple Sobel/central difference gradient on the metric depth map
        grad_x = np.zeros_like(depth_metric)
        grad_y = np.zeros_like(depth_metric)
        grad_x[:, 1:-1] = np.abs(depth_metric[:, 2:] - depth_metric[:, :-2])
        grad_y[1:-1, :] = np.abs(depth_metric[2:, :] - depth_metric[:-2, :])
        grad = grad_x + grad_y
        grad_thresh = float(np.percentile(grad, 85))
        
        # Subsample pixels for 3D back-projection
        ys = np.arange(0, H, subsample)
        xs = np.arange(0, W, subsample)
        xv, yv = np.meshgrid(xs, ys)
        xv, yv = xv.ravel(), yv.ravel()
        depths = depth_metric[yv, xv]
        
        # Filter invalid depths and high gradients
        mask = (depths > 0.5) & (grad[yv, xv] < grad_thresh)
        xv, yv, depths = xv[mask], yv[mask], depths[mask]
        
        if len(depths) == 0:
            continue
            
        # Back-project pixels to camera space and then to world space
        xc = (xv.astype(np.float64) - cx) / fx * depths
        yc = (yv.astype(np.float64) - cy) / fy * depths
        zc = depths.astype(np.float64)
        pts_cam = np.stack([xc, yc, zc], axis=1)
        pts_world = (R_c2w @ pts_cam.T).T + t_c2w
        
        # Filter out points outside max allowed distance
        dist = np.linalg.norm(pts_world - scene_center, axis=1)
        in_bounds = dist < max_dist
        pts_world = pts_world[in_bounds].astype(np.float32)
        colors = img_np[yv[in_bounds], xv[in_bounds]].astype(np.uint8)
        
        all_new_xyz.append(pts_world)
        all_new_rgb.append(colors)
        n_processed += 1
        
    if not all_new_xyz:
        logger.error(f"Depth maps back-projection produced no valid points for scene {scene_dir}.")
        shutil.copy2(ply_backup, ply_path)
        return False
        
    new_xyz = np.concatenate(all_new_xyz, axis=0)
    new_rgb = np.concatenate(all_new_rgb, axis=0)
    
    logger.info(f"Generated {len(new_xyz)} dense points from {n_processed} images")
    
    # Merge existing sparse points with newly generated dense points
    all_xyz = np.concatenate([existing_xyz, new_xyz], axis=0)
    all_rgb = np.concatenate([existing_rgb, new_rgb], axis=0)
    
    # Apply voxel grid downsampling to clean up and unify the point cloud
    logger.info(f"Total points before voxel grid downsampling: {len(all_xyz)}")
    all_xyz_down, all_rgb_down = voxel_down_sample_numpy(all_xyz, all_rgb, voxel_size)
    logger.info(f"Total points after voxel grid downsampling (voxel_size={voxel_size}m): {len(all_xyz_down)}")
    
    # Overwrite the points3D.ply file
    storePly(ply_path, all_xyz_down, all_rgb_down)
    logger.info(f"Successfully saved densified point cloud to {ply_path}")
    return True
