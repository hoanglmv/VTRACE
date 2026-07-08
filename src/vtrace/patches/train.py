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

import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state, get_expon_lr_func
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

try:
    from fused_ssim import fused_ssim
    FUSED_SSIM_AVAILABLE = True
except:
    FUSED_SSIM_AVAILABLE = False

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False

def pearson_correlation_loss(x, y, mask=None):
    if mask is not None:
        x = x[mask]
        y = y[mask]
    else:
        x = x.flatten()
        y = y.flatten()
    
    if len(x) < 3:
        return torch.tensor(0.0, device=x.device)
        
    x_mean = x.mean()
    y_mean = y.mean()
    
    x_diff = x - x_mean
    y_diff = y - y_mean
    
    num = (x_diff * y_diff).sum()
    den = torch.sqrt((x_diff ** 2).sum() * (y_diff ** 2).sum() + 1e-8)
    
    r = num / den
    return 1.0 - r

@torch.no_grad()
def prune_depth_floaters(gaussians, scene, num_cameras_to_check=5, thresh=0.1):
    cams = scene.getTrainCameras()
    if not cams:
        return
    
    import random
    check_cams = random.sample(cams, min(len(cams), num_cameras_to_check))
    
    xyz = gaussians.get_xyz
    N = xyz.shape[0]
    
    floater_votes = torch.zeros(N, dtype=torch.int32, device="cuda")
    overlap_count = torch.zeros(N, dtype=torch.int32, device="cuda")
    
    xyz_homo = torch.cat([xyz, torch.ones((N, 1), device="cuda")], dim=-1)
    
    for cam in check_cams:
        if not cam.depth_reliable or cam.invdepthmap is None:
            continue
            
        pts_w = xyz_homo @ cam.full_proj_transform
        w = pts_w[:, 3:4]
        w = torch.where(w.abs() < 1e-5, torch.ones_like(w) * 1e-5, w)
        pts_ndc = pts_w[:, :3] / w
        
        in_frustum = (pts_ndc[:, 0] >= -1.0) & (pts_ndc[:, 0] <= 1.0) & \
                     (pts_ndc[:, 1] >= -1.0) & (pts_ndc[:, 1] <= 1.0) & \
                     (pts_ndc[:, 2] >= 0.0) & (pts_ndc[:, 2] <= 1.0)
                     
        if not in_frustum.any():
            continue
            
        u = ((pts_ndc[:, 0] + 1.0) * 0.5 * cam.image_width).long()
        v = ((1.0 - pts_ndc[:, 1]) * 0.5 * cam.image_height).long()
        
        u = torch.clamp(u, 0, cam.image_width - 1)
        v = torch.clamp(v, 0, cam.image_height - 1)
        
        pts_cam = xyz_homo @ cam.world_view_transform
        z_cam = pts_cam[:, 2]
        
        inv_z = 1.0 / torch.clamp(z_cam, min=1e-5)
        
        mono_invdepth = cam.invdepthmap[0, v, u]
        depth_mask = cam.depth_mask[0, v, u]
        
        is_floater = in_frustum & (depth_mask > 0) & (inv_z > (mono_invdepth + thresh))
        
        floater_votes += is_floater.int()
        overlap_count += (in_frustum & (depth_mask > 0)).int()
        
    prune_mask = (floater_votes >= 2) & (floater_votes >= (overlap_count // 2))
    
    if prune_mask.any():
        print(f"\n[Depth Pruner] Pruning {prune_mask.sum().item()} floaters out of {N} points.")
        gaussians.prune_points(prune_mask)

def save_loss_history(model_path, history_logs):
    import csv
    csv_file = os.path.join(model_path, "loss_history.csv")
    if not history_logs:
        return
    keys = history_logs[0].keys()
    try:
        with open(csv_file, 'w', newline='') as f:
            dict_writer = csv.DictWriter(f, fieldnames=keys)
            dict_writer.writeheader()
            dict_writer.writerows(history_logs)
        print(f"\n[Loss Logger] Saved training history to {csv_file}")
    except Exception as e:
        print(f"\n[Loss Logger] Failed to save history: {e}")

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    history_logs = []

    if not SPARSE_ADAM_AVAILABLE and opt.optimizer_type == "sparse_adam":
        sys.exit(f"Trying to use sparse adam but it is not installed, please install the correct rasterizer using pip install [3dgs_accel].")

    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    use_sparse_adam = opt.optimizer_type == "sparse_adam" and SPARSE_ADAM_AVAILABLE 
    depth_l1_weight = get_expon_lr_func(opt.depth_l1_weight_init, opt.depth_l1_weight_final, max_steps=opt.iterations)

    viewpoint_stack = scene.getTrainCameras().copy()
    viewpoint_indices = list(range(len(viewpoint_stack)))
    ema_loss_for_log = 0.0
    ema_color_loss_for_log = 0.0
    ema_Ll1depth_for_log = 0.0
    
    # Early stopping variables
    ema_loss_long = None
    loss_history = []

    from gsplat.strategy import MCMCStrategy
    strategy = MCMCStrategy(verbose=False)
    strategy_state = strategy.initialize_state()

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifier=scaling_modifer, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        xyz_lr = gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
            viewpoint_indices = list(range(len(viewpoint_stack)))
        rand_idx = randint(0, len(viewpoint_indices) - 1)
        viewpoint_cam = viewpoint_stack.pop(rand_idx)
        vind = viewpoint_indices.pop(rand_idx)

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        render_pkg = render(viewpoint_cam, gaussians, pipe, bg, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        if viewpoint_cam.alpha_mask is not None:
            alpha_mask = viewpoint_cam.alpha_mask.cuda()
            image *= alpha_mask

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        
        if opt.lambda_edge > 0 and hasattr(viewpoint_cam, 'edge_mask'):
            edge_mask = viewpoint_cam.edge_mask.cuda() # shape (H, W)
            # Spatial weighting map: pixels on edges get more weight
            weight_map = 1.0 + edge_mask.unsqueeze(0) * opt.lambda_edge # shape (1, H, W)
            Ll1 = (torch.abs(image - gt_image) * weight_map).mean()
        else:
            Ll1 = l1_loss(image, gt_image)

        if FUSED_SSIM_AVAILABLE:
            ssim_value = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
        else:
            ssim_value = ssim(image, gt_image)

        color_loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)
        color_loss_val = color_loss.item()
        loss = color_loss

        # Depth regularization
        Ll1depth_pure = 0.0
        if depth_l1_weight(iteration) > 0 and viewpoint_cam.depth_reliable:
            invDepth = render_pkg["depth"]
            mono_invdepth = viewpoint_cam.invdepthmap.cuda()
            depth_mask = viewpoint_cam.depth_mask.cuda()

            Ll1depth_pure = 0.0
            
            # Pearson correlation depth loss to force structure alignment
            p_loss = pearson_correlation_loss(invDepth, mono_invdepth, depth_mask.bool())
            
            # Use only Pearson correlation loss to regularize geometry alignment
            Ll1depth = depth_l1_weight(iteration) * 0.1 * p_loss
            loss += Ll1depth
            Ll1depth = Ll1depth.item()
        else:
            Ll1depth = 0

        # Opacity regularization (Entropy regularization to force opacities to 0 or 1)
        if opt.lambda_opacity > 0:
            eps = 1e-7
            o = gaussians.get_opacity
            entropy = - (o * torch.log(o + eps) + (1.0 - o) * torch.log(1.0 - o + eps))
            opacity_loss = opt.lambda_opacity * entropy.mean()
            loss += opacity_loss

        # Scale regularization (Penalize scaling factors only when they exceed 0.05 to allow normal optimization growth)
        if opt.lambda_scale > 0:
            scale_loss = opt.lambda_scale * torch.clamp(gaussians.get_scaling - 0.05, min=0.0).mean()
            loss += scale_loss

        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_color_loss_for_log = 0.4 * color_loss_val + 0.6 * ema_color_loss_for_log
            ema_Ll1depth_for_log = 0.4 * Ll1depth + 0.6 * ema_Ll1depth_for_log

            # Early stopping check (based purely on visual color reconstruction loss)
            if ema_loss_long is None:
                ema_loss_long = color_loss_val
            else:
                ema_loss_long = 0.999 * ema_loss_long + 0.001 * color_loss_val

            early_stopped = False
            if iteration % 100 == 0:
                loss_history.append((iteration, ema_loss_long))
                if len(loss_history) > 30:
                    loss_history.pop(0)
                
                # Log current metrics to history
                history_logs.append({
                    "iteration": iteration,
                    "loss": loss.item(),
                    "l1_loss": Ll1.item(),
                    "ssim": ssim_value.item() if isinstance(ssim_value, torch.Tensor) else ssim_value,
                    "depth_l1": Ll1depth_pure.item() if isinstance(Ll1depth_pure, torch.Tensor) else Ll1depth_pure,
                    "depth_pearson": p_loss.item() if 'p_loss' in locals() and isinstance(p_loss, torch.Tensor) else 0.0,
                    "opacity_entropy": opacity_loss.item() / opt.lambda_opacity if 'opacity_loss' in locals() and isinstance(opacity_loss, torch.Tensor) and opt.lambda_opacity > 0 else 0.0,
                    "scale_loss": scale_loss.item() if 'scale_loss' in locals() and isinstance(scale_loss, torch.Tensor) else 0.0,
                    "num_points": gaussians.get_xyz.shape[0]
                })
                
                # Periodically backup the loss history to disk
                if iteration % 1000 == 0:
                    save_loss_history(scene.model_path, history_logs)
                
                # Check convergence
                window_size = max(1, opt.early_stopping_window_iters // 100)
                if iteration >= opt.early_stopping_start_iter and len(loss_history) >= window_size:
                    old_iter, old_loss = loss_history[-window_size] # old iterations ago
                    rel_change = abs(ema_loss_long - old_loss) / old_loss
                    if rel_change < opt.early_stopping_rel_change: # relative change over window_size steps
                        print(f"\n[Early Stopping] Converged at iteration {iteration}. Relative change: {rel_change:.5f}")
                        progress_bar.close()
                        scene.save(iteration)
                        save_loss_history(scene.model_path, history_logs)
                        early_stopped = True

            if early_stopped:
                break

            if iteration % 10 == 0:
                progress_bar.set_postfix({
                    "Loss_Tot": f"{ema_loss_for_log:.4f}",
                    "Loss_Col": f"{ema_color_loss_for_log:.4f}"
                })
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background, 1., SPARSE_ADAM_AVAILABLE, None, dataset.train_test_exp), dataset.train_test_exp)
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.exposure_optimizer.step()
                gaussians.exposure_optimizer.zero_grad(set_to_none = True)
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            # MCMC Densification Strategy
            if iteration < opt.densify_until_iter:
                params_dict = {
                    "means": gaussians._xyz,
                    "scales": gaussians._scaling,
                    "quats": gaussians._rotation,
                    "opacities": gaussians._opacity,
                    "f_dc": gaussians._features_dc,
                    "f_rest": gaussians._features_rest
                }

                # Step the strategy
                strategy.step_post_backward(
                    params_dict, 
                    gaussians.optimizers, 
                    strategy_state, 
                    iteration, 
                    {}, 
                    xyz_lr
                )

                # Reassign the tensors back to the model because MCMCStrategy might have recreated them
                gaussians._xyz = params_dict["means"]
                gaussians._scaling = params_dict["scales"]
                gaussians._rotation = params_dict["quats"]
                gaussians._opacity = params_dict["opacities"]
                gaussians._features_dc = params_dict["f_dc"]
                gaussians._features_rest = params_dict["f_rest"]
                
                # Resize helper tensors to match the new number of Gaussians
                num_points = gaussians.get_xyz.shape[0]
                device = gaussians.get_xyz.device
                old_size = gaussians.xyz_gradient_accum.shape[0]
                if num_points != old_size:
                    if num_points > old_size:
                        padding_accum = torch.zeros((num_points - old_size, 1), device=device)
                        gaussians.xyz_gradient_accum = torch.cat([gaussians.xyz_gradient_accum, padding_accum], dim=0)
                        
                        padding_denom = torch.zeros((num_points - old_size, 1), device=device)
                        gaussians.denom = torch.cat([gaussians.denom, padding_denom], dim=0)
                        
                        padding_radii = torch.zeros((num_points - old_size), device=device)
                        gaussians.max_radii2D = torch.cat([gaussians.max_radii2D, padding_radii], dim=0)
                    else:
                        gaussians.xyz_gradient_accum = gaussians.xyz_gradient_accum[:num_points]
                        gaussians.denom = gaussians.denom[:num_points]
                        gaussians.max_radii2D = gaussians.max_radii2D[:num_points]
                
                # Apply depth-guided floater pruning every 500 iterations during the densification phase
                if iteration > 1000 and iteration % 500 == 0:
                    prune_depth_floaters(gaussians, scene, num_cameras_to_check=5, thresh=0.1)
                
            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")
                
    # Save final loss history when training loop finishes
    save_loss_history(scene.model_path, history_logs)

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, train_test_exp):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if train_test_exp:
                        image = image[..., image.shape[-1] // 2:]
                        gt_image = gt_image[..., gt_image.shape[-1] // 2:]
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument('--disable_viewer', action='store_true', default=False)
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    if not args.disable_viewer:
        network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")
