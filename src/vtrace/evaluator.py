"""Local evaluation and submission validation for the VTRACE competition.

The public split contains ground-truth test images, so this module deliberately
uses the exact test_poses.csv manifest instead of scanning arbitrary prediction
files.  Metrics are first averaged inside each scene and scene scores are then
averaged, matching the competition description.
"""

from __future__ import annotations

import csv
import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F


SUPPORTED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


@dataclass(frozen=True)
class ManifestEntry:
    scene: str
    image_name: str
    width: int
    height: int


@dataclass
class ImageMetrics:
    scene: str
    image_name: str
    prediction_path: str
    ground_truth_path: str
    psnr: float
    ssim: float
    lpips: Optional[float]


def read_manifest(
    data_dir: os.PathLike[str] | str,
    scenes: Optional[Sequence[str]] = None,
) -> List[ManifestEntry]:
    data_root = Path(data_dir)
    selected = set(scenes or [])
    entries: List[ManifestEntry] = []
    for scene_dir in sorted(path for path in data_root.iterdir() if path.is_dir()):
        if selected and scene_dir.name not in selected:
            continue
        csv_path = scene_dir / "test" / "test_poses.csv"
        if not csv_path.exists():
            continue
        with csv_path.open("r", newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                entries.append(
                    ManifestEntry(
                        scene=scene_dir.name,
                        image_name=row["image_name"],
                        width=int(row["width"]),
                        height=int(row["height"]),
                    )
                )
    return entries


def _image_files(directory: Path) -> List[Path]:
    if not directory.exists():
        return []
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
    )


def _resolve_image(directory: Path, image_name: str, allow_stem_match: bool) -> Optional[Path]:
    exact = directory / image_name
    if exact.exists():
        return exact
    # Some old local runs changed JPG files to PNG.  Stem matching is useful for
    # measuring those runs, but strict submission validation keeps it disabled.
    if allow_stem_match:
        stem = Path(image_name).stem.casefold()
        matches = [path for path in _image_files(directory) if path.stem.casefold() == stem]
        if len(matches) == 1:
            return matches[0]
    return None


def validate_submission(
    data_dir: os.PathLike[str] | str,
    prediction_dir: os.PathLike[str] | str,
    *,
    strict_names: bool = True,
    scenes: Optional[Sequence[str]] = None,
) -> Dict[str, object]:
    manifest = read_manifest(data_dir, scenes=scenes)
    pred_root = Path(prediction_dir)
    expected = {(entry.scene, entry.image_name) for entry in manifest}
    errors: List[str] = []
    resolved: Dict[Tuple[str, str], str] = {}

    expected_scenes = {entry.scene for entry in manifest}
    actual_scenes = {path.name for path in pred_root.iterdir() if path.is_dir()} if pred_root.exists() else set()
    if scenes:
        actual_scenes &= set(scenes)
    missing_scenes = sorted(expected_scenes - actual_scenes)
    extra_scenes = sorted(actual_scenes - expected_scenes)
    if missing_scenes:
        errors.append(f"Missing scenes: {', '.join(missing_scenes)}")
    if extra_scenes:
        errors.append(f"Extra scenes: {', '.join(extra_scenes)}")

    for entry in manifest:
        path = _resolve_image(pred_root / entry.scene, entry.image_name, not strict_names)
        if path is None:
            errors.append(f"Missing image: {entry.scene}/{entry.image_name}")
            continue
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            errors.append(f"Unreadable image: {path}")
            continue
        height, width = image.shape[:2]
        if (width, height) != (entry.width, entry.height):
            errors.append(
                f"Wrong size: {entry.scene}/{entry.image_name}: "
                f"got {width}x{height}, expected {entry.width}x{entry.height}"
            )
        resolved[(entry.scene, entry.image_name)] = str(path)

    actual = {
        (scene_dir.name, path.name)
        for scene_dir in pred_root.iterdir()
        if scene_dir.is_dir()
        and (not scenes or scene_dir.name in set(scenes))
        for path in _image_files(scene_dir)
    } if pred_root.exists() else set()
    if strict_names:
        for scene, name in sorted(actual - expected):
            errors.append(f"Extra image: {scene}/{name}")

    return {
        "valid": not errors,
        "expected_images": len(manifest),
        "resolved_images": len(resolved),
        "errors": errors,
        "resolved": resolved,
    }


def _load_rgb(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def psnr(prediction: np.ndarray, target: np.ndarray) -> float:
    prediction_f = prediction.astype(np.float64) / 255.0
    target_f = target.astype(np.float64) / 255.0
    mse = float(np.mean((prediction_f - target_f) ** 2))
    if mse == 0.0:
        return float("inf")
    return -10.0 * math.log10(mse)


def ssim(prediction: np.ndarray, target: np.ndarray) -> float:
    """Wang et al. SSIM with an 11x11 Gaussian window, averaged over RGB."""
    pred = prediction.astype(np.float64) / 255.0
    gt = target.astype(np.float64) / 255.0
    c1 = 0.01**2
    c2 = 0.03**2
    channel_scores: List[float] = []
    for channel in range(3):
        x = pred[..., channel]
        y = gt[..., channel]
        mu_x = cv2.GaussianBlur(x, (11, 11), 1.5)
        mu_y = cv2.GaussianBlur(y, (11, 11), 1.5)
        sigma_x = cv2.GaussianBlur(x * x, (11, 11), 1.5) - mu_x * mu_x
        sigma_y = cv2.GaussianBlur(y * y, (11, 11), 1.5) - mu_y * mu_y
        sigma_xy = cv2.GaussianBlur(x * y, (11, 11), 1.5) - mu_x * mu_y
        numerator = (2.0 * mu_x * mu_y + c1) * (2.0 * sigma_xy + c2)
        denominator = (mu_x * mu_x + mu_y * mu_y + c1) * (sigma_x + sigma_y + c2)
        score_map = numerator / np.maximum(denominator, 1e-12)
        # Match the common valid-window implementation.
        channel_scores.append(float(score_map[5:-5, 5:-5].mean()))
    return float(np.mean(channel_scores))


class LPIPSEvaluator:
    def __init__(self, device: str = "auto", network: str = "alex") -> None:
        try:
            import lpips as lpips_package
        except ImportError as exc:
            raise RuntimeError(
                "LPIPS is not installed. Run `uv add lpips` or evaluate with --no-lpips."
            ) from exc
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model = lpips_package.LPIPS(net=network).to(self.device).eval()

    @torch.inference_mode()
    def __call__(self, prediction: np.ndarray, target: np.ndarray) -> float:
        pred = torch.from_numpy(prediction).permute(2, 0, 1).unsqueeze(0).float()
        gt = torch.from_numpy(target).permute(2, 0, 1).unsqueeze(0).float()
        pred = pred.to(self.device) / 127.5 - 1.0
        gt = gt.to(self.device) / 127.5 - 1.0
        return float(self.model(pred, gt, normalize=False).item())


def _mean(values: Iterable[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return float(np.mean(finite)) if finite else float("nan")


def evaluate(
    data_dir: os.PathLike[str] | str,
    prediction_dir: os.PathLike[str] | str,
    *,
    psnr_max: float,
    compute_lpips: bool = True,
    lpips_network: str = "alex",
    device: str = "auto",
    allow_stem_match: bool = False,
    scenes: Optional[Sequence[str]] = None,
) -> Tuple[List[ImageMetrics], Dict[str, Dict[str, float]], Dict[str, float]]:
    data_root = Path(data_dir)
    pred_root = Path(prediction_dir)
    lpips_metric = LPIPSEvaluator(device=device, network=lpips_network) if compute_lpips else None
    image_results: List[ImageMetrics] = []

    for entry in read_manifest(data_root, scenes=scenes):
        prediction_path = _resolve_image(
            pred_root / entry.scene, entry.image_name, allow_stem_match=allow_stem_match
        )
        target_path = _resolve_image(
            data_root / entry.scene / "test" / "images",
            entry.image_name,
            allow_stem_match=True,
        )
        if prediction_path is None:
            raise FileNotFoundError(f"Missing prediction: {entry.scene}/{entry.image_name}")
        if target_path is None:
            raise FileNotFoundError(f"Missing ground truth: {entry.scene}/{entry.image_name}")
        prediction = _load_rgb(prediction_path)
        target = _load_rgb(target_path)
        expected_shape = (entry.height, entry.width, 3)
        if prediction.shape != expected_shape:
            raise ValueError(
                f"Wrong prediction shape for {entry.scene}/{entry.image_name}: "
                f"{prediction.shape}, expected {expected_shape}"
            )
        if target.shape != expected_shape:
            raise ValueError(
                f"Wrong target shape for {entry.scene}/{entry.image_name}: "
                f"{target.shape}, expected {expected_shape}"
            )
        image_results.append(
            ImageMetrics(
                scene=entry.scene,
                image_name=entry.image_name,
                prediction_path=str(prediction_path),
                ground_truth_path=str(target_path),
                psnr=psnr(prediction, target),
                ssim=ssim(prediction, target),
                lpips=lpips_metric(prediction, target) if lpips_metric else None,
            )
        )

    by_scene: Dict[str, List[ImageMetrics]] = {}
    for result in image_results:
        by_scene.setdefault(result.scene, []).append(result)

    scene_results: Dict[str, Dict[str, float]] = {}
    for scene, results in sorted(by_scene.items()):
        mean_psnr = _mean(result.psnr for result in results)
        mean_ssim = _mean(result.ssim for result in results)
        mean_lpips = _mean(
            result.lpips for result in results if result.lpips is not None
        ) if compute_lpips else float("nan")
        psnr_norm = float(np.clip(mean_psnr / psnr_max, 0.0, 1.0))
        score = (
            0.4 * (1.0 - mean_lpips) + 0.3 * mean_ssim + 0.3 * psnr_norm
            if compute_lpips
            else float("nan")
        )
        scene_results[scene] = {
            "images": float(len(results)),
            "psnr": mean_psnr,
            "ssim": mean_ssim,
            "lpips": mean_lpips,
            "psnr_norm": psnr_norm,
            "score": score,
        }

    summary = {
        "scenes": float(len(scene_results)),
        "images": float(len(image_results)),
        "psnr": _mean(result["psnr"] for result in scene_results.values()),
        "ssim": _mean(result["ssim"] for result in scene_results.values()),
        "lpips": _mean(result["lpips"] for result in scene_results.values()),
        "psnr_norm": _mean(result["psnr_norm"] for result in scene_results.values()),
        "score": _mean(result["score"] for result in scene_results.values()),
        "psnr_max": float(psnr_max),
    }
    return image_results, scene_results, summary


def write_evaluation(
    output_dir: os.PathLike[str] | str,
    image_results: Sequence[ImageMetrics],
    scene_results: Dict[str, Dict[str, float]],
    summary: Dict[str, float],
) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    image_fields = list(asdict(image_results[0]).keys()) if image_results else [
        "scene", "image_name", "prediction_path", "ground_truth_path", "psnr", "ssim", "lpips"
    ]
    with (out / "metrics_per_image.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=image_fields)
        writer.writeheader()
        writer.writerows(asdict(result) for result in image_results)
    with (out / "metrics_per_scene.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["scene", "images", "psnr", "ssim", "lpips", "psnr_norm", "score"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for scene, metrics in scene_results.items():
            writer.writerow({"scene": scene, **metrics})
    with (out / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, allow_nan=True)
