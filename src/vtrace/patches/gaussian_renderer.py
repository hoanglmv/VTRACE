#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import math
from gsplat.rendering import rasterization
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh

def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, separate_sh = False, override_color = None, use_trained_exp=False):
    """
    Render the scene using gsplat. 
    
    Background tensor (bg_color) must be on GPU!
    """

    width = int(viewpoint_camera.image_width)
    height = int(viewpoint_camera.image_height)
    
    # Calculate camera intrinsics (K) from FoV
    fx = width / (2 * math.tan(viewpoint_camera.FoVx * 0.5))
    fy = height / (2 * math.tan(viewpoint_camera.FoVy * 0.5))
    cx = width / 2.0
    cy = height / 2.0

    K = torch.tensor([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ], dtype=torch.float32, device="cuda")

    # The viewmat is the world-to-camera matrix.
    # In INRIA, world_view_transform is transposed, so we transpose it back to standard row-major format
    viewmat = viewpoint_camera.world_view_transform.transpose(0, 1)

    means3D = pc.get_xyz
    opacity = pc.get_opacity
    scales = pc.get_scaling * scaling_modifier
    quats = pc.get_rotation

    shs = pc.get_features

    # Handle packed or sparse layout as required by gsplat
    # To keep compatibility with original INRIA code which requires gradients on 2D means:
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Extract radial distortion parameters if available (e.g. SIMPLE_RADIAL from COLMAP)
    radial_coeffs = viewpoint_camera.radial_coeffs.unsqueeze(0) if hasattr(viewpoint_camera, 'radial_coeffs') and viewpoint_camera.radial_coeffs is not None else None

    # gsplat's rasterization function
    render_colors, render_alphas, meta = rasterization(
        means=means3D,
        quats=quats,
        scales=scales,
        opacities=opacity.squeeze(-1),
        colors=shs,
        viewmats=viewmat.unsqueeze(0), # Add batch dimension C=1
        Ks=K.unsqueeze(0),             # Add batch dimension C=1
        width=width,
        height=height,
        near_plane=0.01,
        far_plane=1e10,
        sh_degree=pc.active_sh_degree,
        packed=False,
        backgrounds=bg_color.unsqueeze(0),
        render_mode="RGB+ED",
        rasterize_mode="antialiased" if pipe.antialiasing else "classic",
        radial_coeffs=radial_coeffs
    )

    # Convert back from [1, H, W, 4] to [3, H, W] and [1, H, W] to match INRIA's output format
    rendered_image = render_colors[0, ..., :3].permute(2, 0, 1)
    depth_image = render_colors[0, ..., 3:4].permute(2, 0, 1)
    
    # Scale depth back to camera space units (gsplat output depth is along z-axis, which is correct)
    
    radii = meta["radii"].squeeze(0) # Radii per gaussian

    # Note: screenspace_points needs to be populated with 2D means if used in densification.
    # We set them to meta["means2d"] so we can get gradients.
    screenspace_points = meta["means2d"].squeeze(0)

    # Apply exposure if necessary (kept for VTRACE specific features)
    if use_trained_exp:
        exposure = pc.get_exposure_from_name(viewpoint_camera.image_name)
        rendered_image = torch.exp(exposure.unsqueeze(1).unsqueeze(2)) * rendered_image

    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii,
            "depth": depth_image,
            "meta": meta}
