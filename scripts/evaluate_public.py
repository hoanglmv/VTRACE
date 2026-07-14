#!/usr/bin/env python3
"""Evaluate a VTRACE submission directory against public ground truth."""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.vtrace.evaluator import evaluate, validate_submission, write_evaluation


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="VAI_NVS_DATA/phase1/public_set")
    parser.add_argument("--prediction-dir", required=True)
    parser.add_argument("--output-dir", default="evaluation")
    parser.add_argument("--psnr-max", type=float, default=40.0)
    parser.add_argument("--lpips-network", choices=["alex", "vgg", "squeeze"], default="alex")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-lpips", action="store_true")
    parser.add_argument(
        "--allow-stem-match",
        action="store_true",
        help="Evaluate old runs that changed JPG/JPEG files to PNG. Submission validation remains non-strict.",
    )
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--scene", action="append", dest="scenes", help="Evaluate only selected scenes")
    args = parser.parse_args()

    validation = validate_submission(
        args.data_dir,
        args.prediction_dir,
        strict_names=not args.allow_stem_match,
        scenes=args.scenes,
    )
    printable = {key: value for key, value in validation.items() if key != "resolved"}
    print(json.dumps(printable, indent=2))
    if args.validate_only:
        return 0 if validation["valid"] else 2
    if not validation["valid"] and not args.allow_stem_match:
        print("Submission is invalid; use --allow-stem-match only for legacy local runs.", file=sys.stderr)
        return 2

    image_results, scene_results, summary = evaluate(
        args.data_dir,
        args.prediction_dir,
        psnr_max=args.psnr_max,
        compute_lpips=not args.no_lpips,
        lpips_network=args.lpips_network,
        device=args.device,
        allow_stem_match=args.allow_stem_match,
        scenes=args.scenes,
    )
    write_evaluation(args.output_dir, image_results, scene_results, summary)
    for scene, metrics in scene_results.items():
        print(
            f"{scene}: PSNR={metrics['psnr']:.4f} SSIM={metrics['ssim']:.6f} "
            f"LPIPS={metrics['lpips']:.6f} Score={metrics['score']:.6f}"
        )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
