#!/usr/bin/env python3
"""Audit COLMAP/test intrinsics and distortion round-trip errors."""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src", "vtrace", "gaussian-splatting")))

from scene.colmap_loader import read_intrinsics_binary, read_intrinsics_text


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    args = parser.parse_args()

    failed = False
    for scene_dir in sorted(path for path in Path(args.data_dir).iterdir() if path.is_dir()):
        sparse = scene_dir / "train" / "sparse" / "0"
        cameras = read_intrinsics_binary(str(sparse / "cameras.bin")) if (sparse / "cameras.bin").exists() else read_intrinsics_text(str(sparse / "cameras.txt"))
        if len(cameras) != 1:
            print(f"{scene_dir.name}: WARNING expected one camera, got {len(cameras)}")
        camera = next(iter(cameras.values()))
        if camera.model not in {"SIMPLE_RADIAL", "RADIAL", "PINHOLE", "SIMPLE_PINHOLE", "OPENCV"}:
            print(f"{scene_dir.name}: unsupported camera model {camera.model}")
            failed = True
            continue
        if camera.model in {"SIMPLE_RADIAL", "RADIAL"}:
            fx = fy = float(camera.params[0]); cx, cy = map(float, camera.params[1:3])
            k1 = float(camera.params[3]); k2 = float(camera.params[4]) if len(camera.params) > 4 else 0.0
        elif camera.model == "SIMPLE_PINHOLE":
            fx = fy = float(camera.params[0]); cx, cy = map(float, camera.params[1:3]); k1 = k2 = 0.0
        else:
            fx, fy, cx, cy = map(float, camera.params[:4])
            k1 = float(camera.params[4]) if len(camera.params) > 4 else 0.0
            k2 = float(camera.params[5]) if len(camera.params) > 5 else 0.0
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], np.float64)
        dist = np.array([k1, k2, 0, 0, 0], np.float64)
        grid_x, grid_y = np.meshgrid(np.linspace(0, camera.width - 1, 17), np.linspace(0, camera.height - 1, 13))
        distorted = np.stack([grid_x.ravel(), grid_y.ravel()], axis=-1).reshape(-1, 1, 2)
        undistorted = cv2.undistortPoints(distorted, K, dist)
        redistorted, _ = cv2.projectPoints(
            np.concatenate([undistorted[:, 0], np.ones((len(undistorted), 1))], axis=1),
            np.zeros(3), np.zeros(3), K, dist,
        )
        roundtrip = np.linalg.norm(redistorted[:, 0] - distorted[:, 0], axis=1)

        csv_path = scene_dir / "test" / "test_poses.csv"
        with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        intrinsic_error = max(
            max(abs(float(row["fx"]) - fx), abs(float(row["fy"]) - fy), abs(float(row["cx"]) - cx), abs(float(row["cy"]) - cy))
            for row in rows
        )
        print(
            f"{scene_dir.name}: model={camera.model} k1={k1:+.7f} "
            f"intrinsic_max_error={intrinsic_error:.3e}px "
            f"distortion_roundtrip_p99={np.percentile(roundtrip, 99):.3e}px"
        )
        failed |= intrinsic_error > 1e-4 or np.percentile(roundtrip, 99) > 0.05
    return 2 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

