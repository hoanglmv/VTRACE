#!/usr/bin/env python3
"""Run the pinned max-quality 3DGRUT/3DGUT-MCMC/NHT VTRACE pipeline."""

from __future__ import annotations

import argparse
import copy
import logging
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.vtrace.evaluator import validate_submission
from src.vtrace.nht_adapter import (
    PINNED_3DGRUT_COMMIT,
    SceneRun,
    atomic_write_json,
    atomic_write_text,
    build_train_command,
    create_size_limited_archive,
    discover_scenes,
    find_latest_checkpoint,
    framework_environment,
    framework_python,
    load_config,
    preflight,
    prepare_test_colmap,
    raw_render_set_is_complete,
    render_command,
    run_logged_process,
    utc_now,
    validate_scene,
    verify_framework,
    write_run_manifest,
)


LOGGER = logging.getLogger("vtrace.nht_max")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Resumable max-quality VTRACE training with official NVIDIA 3DGRUT + NHT"
    )
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config" / "nht_max.yaml")
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--framework-dir", type=Path)
    parser.add_argument("--scene", action="append", dest="scenes", help="Repeat to select scenes")
    parser.add_argument("--from-stage", choices=["train", "render"], default="train")
    parser.add_argument("--skip-completed", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--max-retries", type=int)
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run 10 iterations/100k primitives on one scene to compile and verify the server before the paid full run",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = args.config.resolve()
    config = copy.deepcopy(load_config(config_path))
    if args.smoke_test:
        if not args.scenes or len(args.scenes) != 1:
            raise ValueError("--smoke-test requires exactly one --scene")
        config["training"]["iterations"] = 10
        config["training"]["max_gaussians"] = 100000
        config["training"]["checkpoint_iterations"] = [10]
        config["training"]["overrides"] = [
            value
            for value in config["training"].get("overrides", [])
            if not str(value).startswith("model.nht_decoder.color_refine_steps=")
        ] + ["model.nht_decoder.color_refine_steps=0"]
        config["framework"]["minimum_gpu_memory_gib"] = 20
        config["pipeline"]["minimum_free_disk_gib"] = 10
    framework_cfg = config["framework"]
    pipeline_cfg = config["pipeline"]
    training_cfg = config["training"]
    render_cfg = config["render"]

    data_dir = (args.data_dir or Path(pipeline_cfg["data_dir"])).expanduser().resolve()
    default_output = Path("./output_nht_smoke") if args.smoke_test else Path(pipeline_cfg["output_dir"])
    output_dir = (args.output_dir or default_output).expanduser().resolve()
    framework_dir = (args.framework_dir or Path(framework_cfg["directory"])).expanduser().resolve()
    models_dir = output_dir / "models"
    adapters_dir = output_dir / "adapters"
    raw_renders_dir = output_dir / "raw_renders"
    submission_dir = output_dir / "submission"
    logs_dir = output_dir / "logs"
    status_dir = output_dir / "scenes"
    for path in (models_dir, adapters_dir, raw_renders_dir, submission_dir, logs_dir, status_dir):
        path.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(logs_dir / "orchestrator.log", encoding="utf-8")],
    )

    scenes = discover_scenes(data_dir, args.scenes)
    for scene in scenes:
        stats = validate_scene(scene)
        LOGGER.info("Scene %s: %d train images, %d test poses", scene.name, stats["train_images"], stats["test_poses"])

    if args.validate_only:
        validation = validate_submission(data_dir, submission_dir, strict_names=True, scenes=args.scenes)
        serializable = dict(validation)
        serializable.pop("resolved", None)
        atomic_write_json(output_dir / "submission_validation.json", serializable)
        if not validation["valid"]:
            for error in validation["errors"][:50]:
                LOGGER.error("%s", error)
            return 2
        LOGGER.info("Submission is valid: %d images", validation["resolved_images"])
        return 0

    expected_commit = str(framework_cfg.get("commit", PINNED_3DGRUT_COMMIT))
    preflight_info = None
    if args.dry_run:
        if framework_dir.exists():
            verify_framework(framework_dir, expected_commit)
    else:
        preflight_info = preflight(
            framework_dir,
            output_dir,
            expected_commit=expected_commit,
            minimum_gpu_memory_gib=float(framework_cfg.get("minimum_gpu_memory_gib", 22)),
            minimum_gpu_free_gib=float(framework_cfg.get("minimum_gpu_free_gib", 20)),
            minimum_free_disk_gib=float(pipeline_cfg.get("minimum_free_disk_gib", 100)),
        )
    python = framework_python(framework_dir, framework_cfg.get("python"))
    process_environment = framework_environment(framework_dir)
    if preflight_info is not None:
        process_environment["CUDA_VISIBLE_DEVICES"] = str(preflight_info["selected_gpu_index"])
    write_run_manifest(output_dir, config_path, config, scenes, preflight_info)

    retries = int(args.max_retries if args.max_retries is not None else pipeline_cfg.get("max_retries", 1))
    completed_scenes: list[str] = []
    for scene_dir in scenes:
        name = scene_dir.name
        scene_run = SceneRun(name=name, scene_dir=scene_dir, root=status_dir / name)
        scene_run.root.mkdir(parents=True, exist_ok=True)
        model_dir = models_dir / name
        train_done = scene_run.root / "TRAIN_DONE"
        render_done = scene_run.root / "RENDER_DONE"
        existing_checkpoint = find_latest_checkpoint(model_dir)
        train_marker_valid = (
            train_done.exists()
            and existing_checkpoint is not None
            and existing_checkpoint.name == "ckpt_last.pt"
        )
        render_marker_valid = False
        if render_done.exists():
            render_marker_valid = bool(
                validate_submission(data_dir, submission_dir, strict_names=True, scenes=[name])["valid"]
            ) and raw_render_set_is_complete(raw_renders_dir / name, adapters_dir / name)

        if args.from_stage == "train" and not (args.skip_completed and train_marker_valid):
            resume = existing_checkpoint
            if resume is not None and resume.name == "ckpt_last.pt" and args.skip_completed:
                LOGGER.info("Recovered completed training for %s from %s", name, resume)
                if not args.dry_run:
                    atomic_write_text(train_done, f"{utc_now()}\n{resume}\n")
                    scene_run.update("train", "completed", checkpoint=str(resume), recovered=True)
                command = None
            else:
                command = build_train_command(
                    python,
                    framework_dir,
                    scene_dir,
                    models_dir,
                    name,
                    training_cfg,
                    resume,
                )
            if command is not None:
                LOGGER.info("Training %s%s", name, f" (resume {resume.name})" if resume else "")
            if args.dry_run and command is not None:
                LOGGER.info("DRY RUN: %s", " ".join(command))
            elif command is not None:
                for attempt in range(retries + 1):
                    try:
                        run_logged_process(
                            command,
                            cwd=framework_dir,
                            log_path=logs_dir / f"{name}.train.log",
                            scene_run=scene_run,
                            stage="train",
                            env=process_environment,
                        )
                        break
                    except subprocess.CalledProcessError:
                        if attempt >= retries:
                            raise
                        resume = find_latest_checkpoint(model_dir)
                        if resume is None:
                            raise
                        command = build_train_command(
                            python, framework_dir, scene_dir, models_dir, name, training_cfg, resume
                        )
                        LOGGER.warning("Retry %d/%d for %s from %s", attempt + 1, retries, name, resume)
                checkpoint = find_latest_checkpoint(model_dir)
                if checkpoint is None or checkpoint.name != "ckpt_last.pt":
                    raise RuntimeError(f"Training exited successfully but final checkpoint is missing below: {model_dir}")
                atomic_write_text(train_done, f"{utc_now()}\n{checkpoint}\n")
                scene_run.update("train", "completed", checkpoint=str(checkpoint))
        else:
            LOGGER.info("Skipping completed training for %s", name)

        checkpoint = find_latest_checkpoint(model_dir)
        if checkpoint is None:
            if args.dry_run:
                checkpoint = model_dir / "ckpt_last.pt"
            else:
                raise FileNotFoundError(f"No checkpoint available for render: {model_dir}")

        if not (args.skip_completed and render_marker_valid):
            adapter_dir, _ = prepare_test_colmap(scene_dir, adapters_dir / name)
            command = render_command(
                python,
                PROJECT_ROOT,
                framework_dir,
                checkpoint,
                scene_dir,
                adapter_dir,
                raw_renders_dir / name,
                submission_dir / name,
                int(render_cfg.get("jpeg_quality", 100)),
            )
            LOGGER.info("Rendering %s from %s", name, checkpoint.name)
            if args.dry_run:
                LOGGER.info("DRY RUN: %s", " ".join(command))
            else:
                run_logged_process(
                    command,
                    cwd=framework_dir,
                    log_path=logs_dir / f"{name}.render.log",
                    scene_run=scene_run,
                    stage="render",
                    env=process_environment,
                )
                atomic_write_text(render_done, f"{utc_now()}\n{checkpoint}\n")
                scene_run.update("render", "completed", checkpoint=str(checkpoint))
        else:
            LOGGER.info("Skipping completed render for %s", name)
        completed_scenes.append(name)

    if args.dry_run:
        LOGGER.info("Dry run complete for %d scenes", len(scenes))
        return 0

    validation = validate_submission(
        data_dir,
        submission_dir,
        strict_names=True,
        scenes=[scene.name for scene in scenes] if args.scenes else None,
    )
    serializable = dict(validation)
    serializable.pop("resolved", None)
    atomic_write_json(output_dir / "submission_validation.json", serializable)
    if not validation["valid"]:
        raise RuntimeError("Strict submission validation failed; see submission_validation.json")

    archive_name = str(pipeline_cfg.get("archive_name", "submission.zip"))
    archive_path = output_dir / (archive_name if archive_name.lower().endswith(".zip") else archive_name + ".zip")
    max_archive_bytes = int(float(pipeline_cfg.get("max_archive_mb", 350)) * 1_000_000)
    target_archive_bytes = int(float(pipeline_cfg.get("archive_target_mb", 345)) * 1_000_000)
    packaging = create_size_limited_archive(
        raw_renders_dir,
        adapters_dir,
        submission_dir,
        archive_path,
        [scene.name for scene in scenes],
        max_bytes=max_archive_bytes,
        target_bytes=target_archive_bytes,
        maximum_quality=int(render_cfg.get("jpeg_quality", 100)),
    )
    atomic_write_json(output_dir / "packaging.json", packaging)

    # Encoding changes bytes but must never change names or dimensions.
    validation = validate_submission(
        data_dir,
        submission_dir,
        strict_names=True,
        scenes=[scene.name for scene in scenes] if args.scenes else None,
    )
    if not validation["valid"]:
        raise RuntimeError("Submission became invalid during adaptive JPEG packaging")
    atomic_write_json(
        output_dir / "DONE.json",
        {
            "completed_at": utc_now(),
            "scenes": completed_scenes,
            "submission": str(archive_path),
            "submission_bytes": packaging["bytes"],
            "submission_sha256": packaging["sha256"],
            "jpeg_quality": packaging["jpeg_quality"],
            "jpeg_subsampling": packaging["jpeg_subsampling"],
            "images": validation["resolved_images"],
        },
    )
    LOGGER.info("Completed max-quality NHT pipeline: %s", archive_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
