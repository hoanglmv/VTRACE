"""Per-image affine appearance interpolation for unseen camera poses."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np


def qvec2rotmat(qvec: np.ndarray) -> np.ndarray:
    w, x, y, z = qvec
    return np.array(
        [
            [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * z * x + 2 * w * y],
            [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
            [2 * z * x - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y],
        ],
        dtype=np.float64,
    )


def camera_signature(qvec: np.ndarray, tvec: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    rotation = qvec2rotmat(np.asarray(qvec, dtype=np.float64))
    center = -rotation.T @ np.asarray(tvec, dtype=np.float64)
    forward = rotation.T @ np.array([0.0, 0.0, 1.0], dtype=np.float64)
    forward /= np.linalg.norm(forward) + 1e-12
    return center, forward


def _frame_index(name: str) -> Optional[int]:
    match = re.search(r"_(\d{4})_V", name)
    return int(match.group(1)) if match else None


def load_exposure_matrices(model_path: str | Path) -> Dict[str, np.ndarray]:
    path = Path(model_path) / "exposure.json"
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    matrices = {name: np.asarray(value, dtype=np.float32) for name, value in raw.items()}
    return {name: value for name, value in matrices.items() if value.shape == (3, 4)}


def interpolate_exposure(
    target_name: str,
    target_qvec: np.ndarray,
    target_tvec: np.ndarray,
    training_poses: Dict[str, Tuple[np.ndarray, np.ndarray]],
    exposures: Dict[str, np.ndarray],
    *,
    neighbors: int = 8,
    temperature: float = 0.35,
    frame_weight: float = 0.05,
) -> Optional[np.ndarray]:
    candidates = []
    target_center, target_forward = camera_signature(target_qvec, target_tvec)
    target_frame = _frame_index(target_name)
    centers = [camera_signature(*training_poses[name])[0] for name in training_poses if name in exposures]
    if not centers:
        return None
    centers_np = np.stack(centers)
    center_scale = float(np.median(np.linalg.norm(centers_np - np.median(centers_np, axis=0), axis=1)))
    center_scale = max(center_scale, 1e-6)

    for name, (qvec, tvec) in training_poses.items():
        if name not in exposures:
            continue
        center, forward = camera_signature(qvec, tvec)
        position_distance = np.linalg.norm(center - target_center) / center_scale
        angle = math.acos(float(np.clip(np.dot(forward, target_forward), -1.0, 1.0)))
        frame = _frame_index(name)
        frame_distance = abs(frame - target_frame) if frame is not None and target_frame is not None else 0
        score = position_distance + 2.0 * angle + frame_weight * min(frame_distance, 20)
        candidates.append((score, exposures[name]))
    if not candidates:
        return None
    selected = sorted(candidates, key=lambda item: item[0])[: max(1, neighbors)]
    scores = np.asarray([item[0] for item in selected], dtype=np.float64)
    scores -= scores.min()
    weights = np.exp(-scores / max(temperature, 1e-6))
    weights /= weights.sum()
    return sum(float(weight) * matrix for weight, (_, matrix) in zip(weights, selected)).astype(np.float32)

