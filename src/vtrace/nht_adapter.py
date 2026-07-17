"""Resumable VTRACE orchestration for NVIDIA 3DGRUT + NHT.

The NVIDIA framework is intentionally kept in its own virtual environment.
This module only prepares VTRACE camera data, launches the pinned framework,
and records enough state to safely resume a preempted server job.
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import os
import shutil
import struct
import subprocess
import sys
import time
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import yaml
from PIL import Image


LOGGER = logging.getLogger(__name__)
PINNED_3DGRUT_COMMIT = "a37ef721012dea0f29c0fcfff2d525023b4e854a"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    required = {"framework", "pipeline", "training", "render"}
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"Missing config sections: {', '.join(missing)}")
    return config


def discover_scenes(data_dir: Path, selected: Sequence[str] | None = None) -> list[Path]:
    if not data_dir.is_dir():
        raise FileNotFoundError(f"VTRACE data directory does not exist: {data_dir}")
    wanted = set(selected or [])
    scenes = [
        path
        for path in sorted(data_dir.iterdir())
        if path.is_dir() and (path / "train" / "images").is_dir() and (path / "test" / "test_poses.csv").is_file()
    ]
    if wanted:
        available = {path.name for path in scenes}
        missing = sorted(wanted - available)
        if missing:
            raise ValueError(f"Unknown scenes: {', '.join(missing)}")
        scenes = [path for path in scenes if path.name in wanted]
    if not scenes:
        raise ValueError(f"No valid VTRACE scenes found under {data_dir}")
    return scenes


def validate_scene(scene_dir: Path) -> dict[str, int]:
    sparse = scene_dir / "train" / "sparse" / "0"
    required = [
        scene_dir / "train" / "images",
        sparse / "cameras.bin",
        sparse / "images.bin",
        sparse / "points3D.bin",
        scene_dir / "test" / "test_poses.csv",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Scene is incomplete:\n  " + "\n  ".join(missing))
    train_images = sum(1 for path in required[0].iterdir() if path.is_file())
    with required[-1].open("r", encoding="utf-8-sig", newline="") as handle:
        test_rows = list(csv.DictReader(handle))
    if not test_rows:
        raise ValueError(f"No test poses in {required[-1]}")
    names = [row["image_name"] for row in test_rows]
    if len(names) != len(set(names)):
        raise ValueError(f"Duplicate test image names in {required[-1]}")
    return {"train_images": train_images, "test_poses": len(test_rows)}


CAMERA_MODELS = {
    0: ("SIMPLE_PINHOLE", 3),
    1: ("PINHOLE", 4),
    2: ("SIMPLE_RADIAL", 4),
    3: ("RADIAL", 5),
    4: ("OPENCV", 8),
    5: ("OPENCV_FISHEYE", 8),
    6: ("FULL_OPENCV", 12),
}


def read_first_colmap_camera(path: Path) -> dict[str, Any]:
    """Read the first camera from a COLMAP cameras.bin file."""
    with path.open("rb") as handle:
        count_data = handle.read(8)
        if len(count_data) != 8:
            raise ValueError(f"Invalid COLMAP camera file: {path}")
        count = struct.unpack("<Q", count_data)[0]
        if count < 1:
            raise ValueError(f"COLMAP camera file is empty: {path}")
        camera_id, model_id, width, height = struct.unpack("<iiQQ", handle.read(24))
        if model_id not in CAMERA_MODELS:
            raise ValueError(f"Unsupported COLMAP camera model id {model_id} in {path}")
        model, parameter_count = CAMERA_MODELS[model_id]
        params = struct.unpack("<" + "d" * parameter_count, handle.read(8 * parameter_count))
    return {
        "id": camera_id,
        "model": model,
        "width": width,
        "height": height,
        "params": params,
    }


def distortion_from_camera(camera: dict[str, Any]) -> tuple[float, float, float, float]:
    model = camera["model"]
    params = camera["params"]
    if model == "SIMPLE_RADIAL":
        return float(params[3]), 0.0, 0.0, 0.0
    if model == "RADIAL":
        return float(params[3]), float(params[4]), 0.0, 0.0
    if model in {"OPENCV", "OPENCV_FISHEYE", "FULL_OPENCV"}:
        return float(params[4]), float(params[5]), float(params[6]), float(params[7])
    return 0.0, 0.0, 0.0, 0.0


def prepare_test_colmap(scene_dir: Path, adapter_dir: Path) -> tuple[Path, list[str]]:
    """Create a test-only COLMAP dataset consumed by the official renderer.

    Test images are black placeholders. They are used only because the official
    data loader expects an image for every camera; predictions never depend on
    their content.
    """
    csv_path = scene_dir / "test" / "test_poses.csv"
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    camera = read_first_colmap_camera(scene_dir / "train" / "sparse" / "0" / "cameras.bin")
    k1, k2, p1, p2 = distortion_from_camera(camera)

    images_dir = adapter_dir / "images"
    sparse_dir = adapter_dir / "sparse" / "0"
    images_dir.mkdir(parents=True, exist_ok=True)
    sparse_dir.mkdir(parents=True, exist_ok=True)

    camera_lines = ["# Camera list generated from VTRACE test_poses.csv"]
    image_lines = ["# Image list generated from VTRACE test_poses.csv"]
    names: list[str] = []
    camera_ids: dict[tuple[int, int, float, float, float, float], int] = {}
    for index, row in enumerate(rows, start=1):
        name = row["image_name"]
        names.append(name)
        width, height = int(row["width"]), int(row["height"])
        fx, fy = float(row["fx"]), float(row["fy"])
        cx, cy = float(row["cx"]), float(row["cy"])
        intrinsics_key = (width, height, fx, fy, cx, cy)
        camera_id = camera_ids.get(intrinsics_key)
        if camera_id is None:
            camera_id = len(camera_ids) + 1
            camera_ids[intrinsics_key] = camera_id
            if abs(fx - fy) <= 1e-8 and p1 == 0.0 and p2 == 0.0 and k2 == 0.0:
                camera_lines.append(
                    f"{camera_id} SIMPLE_RADIAL {width} {height} {fx:.17g} {cx:.17g} {cy:.17g} {k1:.17g}"
                )
            else:
                camera_lines.append(
                    f"{camera_id} OPENCV {width} {height} {fx:.17g} {fy:.17g} {cx:.17g} {cy:.17g} "
                    f"{k1:.17g} {k2:.17g} {p1:.17g} {p2:.17g}"
                )
        pose = " ".join(
            f"{float(row[key]):.17g}"
            for key in ("qw", "qx", "qy", "qz", "tx", "ty", "tz")
        )
        image_lines.extend([f"{index} {pose} {camera_id} {name}", ""])

        image_path = images_dir / name
        if not image_path.exists():
            image_path.parent.mkdir(parents=True, exist_ok=True)
            placeholder = Image.new("RGB", (width, height), (0, 0, 0))
            if image_path.suffix.lower() in {".jpg", ".jpeg"}:
                placeholder.save(image_path, "JPEG", quality=75, subsampling=2)
            else:
                placeholder.save(image_path, "PNG", compress_level=9)

    atomic_write_text(sparse_dir / "cameras.txt", "\n".join(camera_lines) + "\n")
    atomic_write_text(sparse_dir / "images.txt", "\n".join(image_lines) + "\n")
    points_source = (scene_dir / "train" / "sparse" / "0" / "points3D.bin").resolve()
    points_target = sparse_dir / "points3D.bin"
    if not points_target.exists():
        os.symlink(points_source, points_target)
    elif points_target.resolve() != points_source:
        raise RuntimeError(f"Stale adapter point cloud at {points_target}; expected {points_source}")
    atomic_write_json(adapter_dir / "manifest.json", {"scene": scene_dir.name, "images": names})
    return adapter_dir, names


def find_latest_checkpoint(scene_model_dir: Path) -> Path | None:
    def is_complete(path: Path) -> bool:
        # Current PyTorch checkpoints are ZIP containers. A provider kill in
        # torch.save commonly leaves a non-empty file without a central directory.
        return path.is_file() and path.stat().st_size > 0 and zipfile.is_zipfile(path)

    completed = [path for path in scene_model_dir.rglob("ckpt_last.pt") if is_complete(path)]
    if completed:
        return max(completed, key=lambda path: path.stat().st_mtime)
    candidates: list[tuple[int, float, Path]] = []
    for path in scene_model_dir.rglob("ckpt_*.pt"):
        if path.name == "ckpt_last.pt":
            continue
        try:
            iteration = int(path.stem.rsplit("_", 1)[1])
        except (IndexError, ValueError):
            continue
        if is_complete(path):
            candidates.append((iteration, path.stat().st_mtime, path))
    return max(candidates, default=(0, 0.0, None), key=lambda item: (item[0], item[1]))[2]


def framework_python(framework_dir: Path, configured: str | None = None) -> Path:
    if configured:
        candidate = Path(configured).expanduser()
        if not candidate.is_absolute():
            candidate = framework_dir / candidate
    else:
        candidate = framework_dir / ".venv" / "bin" / "python"
    if not candidate.is_file():
        raise FileNotFoundError(
            f"3DGRUT Python not found: {candidate}. Run scripts/setup_3dgrut_nht.sh first."
        )
    # A venv's python executable is normally a symlink to the base interpreter.
    # Resolving that symlink bypasses pyvenv.cfg and therefore drops all packages
    # installed in the 3DGRUT environment (for example hydra and ncore).
    return candidate.absolute()


def framework_environment(framework_dir: Path) -> dict[str, str]:
    """Restore the CUDA toolkit selected by setup_all.sh for JIT compilation."""
    environment = os.environ.copy()
    env_file = framework_dir.parent / f".{framework_dir.name}.vtrace-nht-env"
    if not env_file.is_file():
        return environment
    values: dict[str, str] = {}
    for line in env_file.read_text(encoding="utf-8").splitlines():
        key, separator, value = line.partition("=")
        if separator and key in {"CUDA_HOME"} and value:
            values[key] = value
    cuda_home = values.get("CUDA_HOME")
    if cuda_home:
        environment["CUDA_HOME"] = cuda_home
        environment["PATH"] = f"{cuda_home}/bin:{environment.get('PATH', '')}"
        environment["LD_LIBRARY_PATH"] = f"{cuda_home}/lib64:{environment.get('LD_LIBRARY_PATH', '')}"
    return environment


def verify_framework(framework_dir: Path, expected_commit: str = PINNED_3DGRUT_COMMIT) -> str:
    train_script = framework_dir / "train.py"
    nht_config = framework_dir / "configs" / "apps" / "colmap_3dgut_mcmc_nht.yaml"
    if not train_script.is_file() or not nht_config.is_file():
        raise FileNotFoundError(f"3DGRUT/NHT checkout is incomplete: {framework_dir}")
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=framework_dir, text=True, capture_output=True, check=True
    )
    actual = result.stdout.strip()
    if actual != expected_commit:
        raise RuntimeError(
            f"3DGRUT commit mismatch: got {actual}, expected {expected_commit}. "
            "Refusing an unrepeatable max-quality run."
        )
    return actual


def gpu_memory_info_mib() -> list[dict[str, int]]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.total,memory.free",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        capture_output=True,
        check=True,
    )
    memories: list[dict[str, int]] = []
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        index, total, free = (int(value.strip()) for value in line.split(",", maxsplit=2))
        memories.append({"index": index, "total": total, "free": free})
    return memories


def preflight(
    framework_dir: Path,
    output_dir: Path,
    *,
    expected_commit: str,
    minimum_gpu_memory_gib: float,
    minimum_gpu_free_gib: float,
    minimum_free_disk_gib: float,
) -> dict[str, Any]:
    commit = verify_framework(framework_dir, expected_commit)
    memories = gpu_memory_info_mib()
    if not memories:
        raise RuntimeError("nvidia-smi did not report any GPU")
    eligible = [
        memory
        for memory in memories
        if memory["total"] / 1024.0 >= minimum_gpu_memory_gib
        and memory["free"] / 1024.0 >= minimum_gpu_free_gib
    ]
    if not eligible:
        best = max(memories, key=lambda memory: (memory["total"], memory["free"]))
        raise RuntimeError(
            f"NHT 1M requires at least {minimum_gpu_memory_gib:.0f} GiB total and "
            f"{minimum_gpu_free_gib:.0f} GiB free VRAM; best visible GPU has "
            f"{best['total'] / 1024.0:.1f} GiB total/{best['free'] / 1024.0:.1f} GiB free. "
            "Use an idle RTX 3090 24GB or a larger GPU."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    free_gib = shutil.disk_usage(output_dir).free / (1024**3)
    if free_gib < minimum_free_disk_gib:
        raise RuntimeError(
            f"Only {free_gib:.1f} GiB free at {output_dir}; max-quality run requires "
            f"at least {minimum_free_disk_gib:.0f} GiB for checkpoints and renders."
        )
    selected = max(eligible, key=lambda memory: (memory["free"], memory["total"]))
    return {
        "framework_commit": commit,
        "gpu_memory_mib": memories,
        "selected_gpu_index": selected["index"],
        "free_disk_gib": free_gib,
    }


@dataclass
class SceneRun:
    name: str
    scene_dir: Path
    root: Path

    @property
    def status_path(self) -> Path:
        return self.root / "status.json"

    def update(self, stage: str, state: str, **details: Any) -> None:
        payload: dict[str, Any] = {}
        if self.status_path.exists():
            try:
                payload = json.loads(self.status_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                payload = {}
        payload.update({"scene": self.name, "updated_at": utc_now(), "stage": stage, "state": state})
        payload.update(details)
        atomic_write_json(self.status_path, payload)


def run_logged_process(
    command: Sequence[str],
    *,
    cwd: Path,
    log_path: Path,
    scene_run: SceneRun,
    stage: str,
    heartbeat_seconds: float = 20.0,
    env: dict[str, str] | None = None,
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("ab", buffering=0) as log:
        header = f"\n[{utc_now()}] COMMAND: {' '.join(command)}\n".encode()
        log.write(header)
        process = subprocess.Popen(
            list(command),
            cwd=cwd,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=False,
            env=env,
        )
        scene_run.update(stage, "running", pid=process.pid, command=list(command), log=str(log_path))
        while True:
            return_code = process.poll()
            if return_code is not None:
                break
            scene_run.update(stage, "running", pid=process.pid, heartbeat_at=utc_now(), log=str(log_path))
            time.sleep(heartbeat_seconds)
    if return_code != 0:
        scene_run.update(stage, "failed", return_code=return_code, log=str(log_path))
        raise subprocess.CalledProcessError(return_code, command)


def checkpoint_override(iterations: Iterable[int], final_iteration: int) -> str:
    values = sorted({int(value) for value in iterations if 0 < int(value) <= final_iteration} | {final_iteration})
    return "checkpoint.iterations=[" + ",".join(str(value) for value in values) + "]"


def build_train_command(
    python: Path,
    framework_dir: Path,
    scene_dir: Path,
    model_root: Path,
    scene_name: str,
    training: dict[str, Any],
    resume: Path | None,
) -> list[str]:
    iterations = int(training["iterations"])
    command = [
        str(python),
        str(framework_dir / "train.py"),
        "--config-name",
        str(training.get("config_name", "apps/colmap_3dgut_mcmc_nht.yaml")),
        f"path={scene_dir / 'train'}",
        f"out_dir={model_root}",
        f"experiment_name={scene_name}",
        f"n_iterations={iterations}",
        "dataset.downsample_factor=1",
        "dataset.test_split_interval=0",
        "dataset.load_exif=false",
        "test_last=false",
        "compute_extra_metrics=false",
        f"num_workers={int(training.get('num_workers', 16))}",
        f"strategy.add.max_n_gaussians={int(training['max_gaussians'])}",
        checkpoint_override(training.get("checkpoint_iterations", []), iterations),
    ]
    for override in training.get("overrides", []):
        command.append(str(override))
    if resume is not None:
        command.append(f"resume={resume}")
    return command


def render_command(
    python: Path,
    project_root: Path,
    framework_dir: Path,
    checkpoint: Path,
    scene_dir: Path,
    adapter_dir: Path,
    raw_render_dir: Path,
    submission_dir: Path,
    jpeg_quality: int,
) -> list[str]:
    return [
        str(python),
        str(project_root / "scripts" / "3dgrut_render_vtrace.py"),
        "--framework-dir",
        str(framework_dir),
        "--checkpoint",
        str(checkpoint),
        "--train-path",
        str(scene_dir / "train"),
        "--test-path",
        str(adapter_dir),
        "--raw-output",
        str(raw_render_dir),
        "--submission-dir",
        str(submission_dir),
        "--jpeg-quality",
        str(jpeg_quality),
    ]


def materialize_submission_from_raw(
    raw_renders_dir: Path,
    adapters_dir: Path,
    submission_dir: Path,
    scene_names: Sequence[str],
    *,
    jpeg_quality: int,
    jpeg_subsampling: int,
) -> int:
    """Encode submission images from lossless raw renders, never from prior JPEGs."""
    encoded = 0
    for scene_name in scene_names:
        manifest_path = adapters_dir / scene_name / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Missing render manifest: {manifest_path}")
        image_names = json.loads(manifest_path.read_text(encoding="utf-8"))["images"]
        raw_dir = raw_renders_dir / scene_name / "predictions"
        scene_output = submission_dir / scene_name
        scene_output.mkdir(parents=True, exist_ok=True)
        for index, image_name in enumerate(image_names):
            source = raw_dir / f"{index:05d}.png"
            target = scene_output / image_name
            if not source.is_file():
                raise FileNotFoundError(f"Missing lossless render: {source}")
            temporary = target.with_name(target.name + ".tmp")
            with Image.open(source) as image:
                image = image.convert("RGB")
                if target.suffix.lower() in {".jpg", ".jpeg"}:
                    image.save(
                        temporary,
                        "JPEG",
                        quality=int(jpeg_quality),
                        subsampling=int(jpeg_subsampling),
                        optimize=True,
                    )
                else:
                    image.save(temporary, "PNG", optimize=True)
            os.replace(temporary, target)
            encoded += 1
    return encoded


def raw_render_set_is_complete(raw_scene_dir: Path, adapter_dir: Path) -> bool:
    manifest_path = adapter_dir / "manifest.json"
    if not manifest_path.is_file():
        return False
    try:
        image_names = json.loads(manifest_path.read_text(encoding="utf-8"))["images"]
        for index, image_name in enumerate(image_names):
            raw_path = raw_scene_dir / "predictions" / f"{index:05d}.png"
            expected_path = adapter_dir / "images" / image_name
            with Image.open(raw_path) as raw, Image.open(expected_path) as expected:
                if raw.size != expected.size:
                    return False
                raw.verify()
        return True
    except (FileNotFoundError, KeyError, json.JSONDecodeError, OSError, ValueError):
        return False


def create_size_limited_archive(
    raw_renders_dir: Path,
    adapters_dir: Path,
    submission_dir: Path,
    archive_path: Path,
    scene_names: Sequence[str],
    *,
    max_bytes: int,
    target_bytes: int,
    maximum_quality: int = 100,
) -> dict[str, Any]:
    """Choose the highest JPEG quality whose real ZIP stays under target_bytes."""
    if not 0 < target_bytes <= max_bytes:
        raise ValueError("Archive target must be positive and no larger than the hard limit")
    archive_path.parent.mkdir(parents=True, exist_ok=True)

    attempts: list[dict[str, int]] = []

    def attempt(quality: int, subsampling: int) -> int:
        materialize_submission_from_raw(
            raw_renders_dir,
            adapters_dir,
            submission_dir,
            scene_names,
            jpeg_quality=quality,
            jpeg_subsampling=subsampling,
        )
        temporary_archive = archive_path.with_name(archive_path.name + ".tmp")
        with zipfile.ZipFile(
            temporary_archive,
            "w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=6,
        ) as archive:
            for scene_name in scene_names:
                manifest = json.loads(
                    (adapters_dir / scene_name / "manifest.json").read_text(encoding="utf-8")
                )
                for image_name in manifest["images"]:
                    source = submission_dir / scene_name / image_name
                    archive.write(source, arcname=str(Path(scene_name) / image_name))
        os.replace(temporary_archive, archive_path)
        size = archive_path.stat().st_size
        attempts.append({"quality": quality, "subsampling": subsampling, "bytes": size})
        return size

    best: tuple[int, int, int] | None = None
    # Preserve 4:4:4 chroma whenever possible. If even quality 40 cannot fit,
    # fall back to 4:2:0 and repeat the search.
    for subsampling, minimum_quality in ((0, 40), (2, 30)):
        low, high = minimum_quality, int(maximum_quality)
        while low <= high:
            quality = (low + high) // 2
            size = attempt(quality, subsampling)
            if size <= target_bytes:
                best = (quality, subsampling, size)
                low = quality + 1
            else:
                high = quality - 1
        if best is not None:
            break

    if best is None:
        smallest = min(attempts, key=lambda item: item["bytes"])
        raise RuntimeError(
            f"Could not fit submission below {max_bytes} bytes; smallest attempt was "
            f"{smallest['bytes']} bytes at JPEG quality {smallest['quality']}."
        )

    quality, subsampling, _ = best
    final_size = attempt(quality, subsampling)
    if final_size > max_bytes:
        raise RuntimeError(f"Final archive is {final_size} bytes, over hard limit {max_bytes}")
    return {
        "archive": str(archive_path),
        "bytes": final_size,
        "sha256": sha256_file(archive_path),
        "max_bytes": max_bytes,
        "target_bytes": target_bytes,
        "jpeg_quality": quality,
        "jpeg_subsampling": subsampling,
        "attempts": attempts,
    }


def write_run_manifest(
    output_dir: Path,
    config_path: Path,
    config: dict[str, Any],
    scenes: Sequence[Path],
    preflight_info: dict[str, Any] | None,
) -> None:
    manifest = {
        "created_at": utc_now(),
        "config_path": str(config_path.resolve()),
        "config_sha256": sha256_file(config_path),
        "config": config,
        "scenes": [path.name for path in scenes],
        "preflight": preflight_info,
        "python": sys.version,
    }
    atomic_write_json(output_dir / "manifest.json", manifest)
    with (output_dir / "config.resolved.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
