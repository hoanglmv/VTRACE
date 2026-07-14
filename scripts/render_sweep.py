#!/usr/bin/env python3
"""Render controlled rasterizer ablations from existing VTRACE checkpoints."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.vtrace.data_utils import list_scenes
from src.vtrace.evaluator import evaluate, write_evaluation
from src.vtrace.renderer import render_scene


VARIANTS = {
    "native_classic": dict(distortion_mode="native", with_ut=True, antialiasing=False),
    "native_aa": dict(distortion_mode="native", with_ut=True, antialiasing=True),
    "native_no_ut": dict(distortion_mode="native", with_ut=False, antialiasing=False),
    "legacy_remap": dict(distortion_mode="legacy_remap", with_ut=False, antialiasing=False),
    "pinhole": dict(distortion_mode="none", with_ut=False, antialiasing=False),
    "ss125_area": dict(distortion_mode="native", with_ut=True, antialiasing=True, supersample=1.25, downsample_filter="area"),
    "ss150_area": dict(distortion_mode="native", with_ut=True, antialiasing=True, supersample=1.5, downsample_filter="area"),
    "ss200_area": dict(distortion_mode="native", with_ut=True, antialiasing=True, supersample=2.0, downsample_filter="area"),
    "ss150_lanczos": dict(distortion_mode="native", with_ut=True, antialiasing=True, supersample=1.5, downsample_filter="lanczos"),
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--models-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--variants", nargs="+", choices=sorted(VARIANTS), default=list(VARIANTS))
    parser.add_argument("--scenes", nargs="*", default=None)
    parser.add_argument("--checkpoint-iteration", default="latest")
    parser.add_argument("--jpeg-quality", type=int, default=100)
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--no-lpips", action="store_true")
    parser.add_argument("--psnr-max", type=float, default=40.0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    models_dir = Path(args.models_dir)
    output_dir = Path(args.output_dir)
    scenes = args.scenes or list_scenes(str(data_dir))

    sweep_summary = {}
    for variant in args.variants:
        variant_root = output_dir / variant
        submission_dir = variant_root / "submission"
        if submission_dir.exists() and args.overwrite:
            shutil.rmtree(submission_dir)
        options = {"supersample": 1.0, "downsample_filter": "area", **VARIANTS[variant]}
        for scene in scenes:
            render_scene(
                scene,
                str(data_dir / scene),
                str(models_dir / scene),
                str(submission_dir),
                render_format="jpg",
                checkpoint_iteration=args.checkpoint_iteration,
                jpeg_quality=args.jpeg_quality,
                **options,
            )
        if args.evaluate:
            image_results, scene_results, summary = evaluate(
                data_dir,
                submission_dir,
                psnr_max=args.psnr_max,
                compute_lpips=not args.no_lpips,
            )
            write_evaluation(variant_root / "evaluation", image_results, scene_results, summary)
            sweep_summary[variant] = summary
            print(f"{variant}: PSNR={summary['psnr']:.4f} SSIM={summary['ssim']:.6f}")

    if sweep_summary:
        output_dir.mkdir(parents=True, exist_ok=True)
        with (output_dir / "sweep_summary.json").open("w", encoding="utf-8") as handle:
            json.dump(sweep_summary, handle, indent=2, allow_nan=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

