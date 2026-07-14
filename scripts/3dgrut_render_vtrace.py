#!/usr/bin/env python3
"""Render VTRACE test_poses.csv with a normalized 3DGRUT/NHT checkpoint."""

from __future__ import annotations

import argparse
import copy
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image


def save_submission_image(source: Path, target: Path, jpeg_quality: int) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    with Image.open(source) as image:
        image = image.convert("RGB")
        if target.suffix.lower() in {".jpg", ".jpeg"}:
            image.save(temporary, "JPEG", quality=jpeg_quality, subsampling=0)
        else:
            image.save(temporary, "PNG")
    os.replace(temporary, target)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--framework-dir", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--train-path", required=True, type=Path)
    parser.add_argument("--test-path", required=True, type=Path)
    parser.add_argument("--raw-output", required=True, type=Path)
    parser.add_argument("--submission-dir", required=True, type=Path)
    parser.add_argument("--jpeg-quality", type=int, default=100)
    args = parser.parse_args()

    sys.path.insert(0, str(args.framework_dir.resolve()))
    import torch
    import torchvision
    from torch.utils.data import DataLoader
    import threedgrut.datasets as datasets
    from threedgrut.datasets.colmap_gsplat import normalize_world_space, scene_scale, transform_cameras
    from threedgrut.datasets.dataset_colmap import ColmapDataset
    from threedgrut.datasets.utils import configure_dataloader_for_platform
    from threedgrut.render import Renderer
    from threedgrut.utils.render import apply_background, apply_feature_decoder, apply_post_processing

    args.raw_output.mkdir(parents=True, exist_ok=True)
    renderer = Renderer.from_checkpoint(
        checkpoint_path=str(args.checkpoint),
        path=str(args.test_path),
        out_dir=str(args.raw_output),
        save_gt=False,
        computes_extra_metrics=False,
    )

    # The official NHT config normalizes world coordinates from training cameras.
    # A test-only COLMAP model would otherwise compute a different transform.
    conf = copy.deepcopy(renderer.conf)
    conf.path = str(args.test_path)
    conf.dataset.normalize_world_space = False
    conf.dataset.test_split_interval = 0
    conf.dataset.load_exif = False
    test_dataset = datasets.make_test(name=conf.dataset.type, config=conf)

    reference = ColmapDataset(
        str(args.train_path),
        split="train",
        downsample_factor=1,
        test_split_interval=0,
        normalize_world_space=False,
        gsplat_image_downscale=False,
    )
    points = reference._load_points_for_world_normalization()
    _, _, transform = normalize_world_space(reference.poses, points)
    test_dataset.poses = transform_cameras(transform, test_dataset.poses).astype(np.float32)
    test_dataset.camera_centers = test_dataset.poses[:, :3, 3].copy()
    test_dataset.cameras_extent = scene_scale(test_dataset.poses) * 1.1
    test_dataset.center, test_dataset.length_scale, test_dataset.scene_bbox = test_dataset.compute_spatial_extents()

    loader_options = configure_dataloader_for_platform(
        {"num_workers": 4, "batch_size": 1, "shuffle": False, "collate_fn": None}
    )
    renderer.dataset = test_dataset
    renderer.dataloader = DataLoader(test_dataset, **loader_options)
    renderer.save_gt = False
    renderer.compute_extra_metrics = False

    image_names = [camera.name for camera in test_dataset.cam_extrinsics]
    official_dir = args.raw_output / "predictions"
    official_dir.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        for index, batch in enumerate(renderer.dataloader):
            output_path = official_dir / f"{index:05d}.png"
            expected_name = image_names[index]
            expected_path = args.test_path / "images" / expected_name
            if output_path.is_file():
                try:
                    with Image.open(output_path) as existing, Image.open(expected_path) as expected:
                        if existing.size == expected.size:
                            existing.verify()
                            continue
                except (OSError, ValueError):
                    pass

            gpu_batch = renderer.dataset.get_gpu_batch_with_intrinsics(batch)
            outputs = renderer.model(gpu_batch)
            if renderer.feature_decoder is not None:
                outputs = apply_feature_decoder(
                    renderer.feature_decoder,
                    outputs,
                    gpu_batch,
                    training=False,
                    center_ray_encoding=bool(
                        getattr(renderer.conf.model.nht_decoder, "center_ray_encoding", False)
                    ),
                )
            outputs = apply_background(renderer.model.background, outputs, gpu_batch, training=False)
            if renderer.post_processing is not None:
                outputs = apply_post_processing(renderer.post_processing, outputs, gpu_batch, training=False)
            temporary = output_path.with_name(output_path.name + ".tmp")
            torchvision.utils.save_image(
                outputs["pred_features"].squeeze(0).permute(2, 0, 1),
                temporary,
                format="png",
            )
            os.replace(temporary, output_path)

    for index, image_name in enumerate(image_names):
        source = official_dir / f"{index:05d}.png"
        if not source.is_file():
            raise FileNotFoundError(source)
        save_submission_image(source, args.submission_dir / image_name, args.jpeg_quality)

    # Strict decode and dimension check against the adapter placeholders.
    for image_name in image_names:
        expected_path = args.test_path / "images" / image_name
        output_path = args.submission_dir / image_name
        with Image.open(expected_path) as expected, Image.open(output_path) as rendered:
            if rendered.size != expected.size:
                raise ValueError(f"Wrong render size for {image_name}: {rendered.size} != {expected.size}")
            rendered.verify()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
