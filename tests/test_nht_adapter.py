from __future__ import annotations

import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from PIL import Image

from src.vtrace.nht_adapter import (
    checkpoint_override,
    create_size_limited_archive,
    find_latest_checkpoint,
    framework_environment,
    prepare_test_colmap,
    read_first_colmap_camera,
    validate_scene,
)


ROOT = Path(__file__).resolve().parents[1]
PUBLIC_SCENE = ROOT / "VAI_NVS_DATA" / "phase1" / "public_set" / "HCM0181"


@unittest.skipUnless(PUBLIC_SCENE.exists(), "VTRACE public data is not present")
class NHTAdapterTests(unittest.TestCase):
    def test_size_limited_archive_uses_exact_layout(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "raw" / "scene" / "predictions"
            adapter = root / "adapters" / "scene"
            submission = root / "submission"
            raw.mkdir(parents=True)
            adapter.mkdir(parents=True)
            names = [f"frame_{index}.JPG" for index in range(4)]
            (adapter / "manifest.json").write_text(json.dumps({"images": names}))
            for index in range(len(names)):
                noise = Image.effect_noise((256, 256), 100 + index).convert("RGB")
                noise.save(raw / f"{index:05d}.png", "PNG")
            archive_path = root / "submission.zip"
            result = create_size_limited_archive(
                root / "raw",
                root / "adapters",
                submission,
                archive_path,
                ["scene"],
                max_bytes=200_000,
                target_bytes=190_000,
                maximum_quality=100,
            )
            self.assertLessEqual(result["bytes"], 200_000)
            with zipfile.ZipFile(archive_path) as archive:
                self.assertEqual(
                    sorted(archive.namelist()),
                    sorted(f"scene/{name}" for name in names),
                )

    def test_framework_environment_restores_cuda_toolkit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            framework = Path(directory) / "3dgrut"
            framework.mkdir()
            (framework.parent / ".3dgrut.vtrace-nht-env").write_text("CUDA_HOME=/opt/cuda-12.8\n")
            environment = framework_environment(framework)
            self.assertEqual(environment["CUDA_HOME"], "/opt/cuda-12.8")
            self.assertTrue(environment["PATH"].startswith("/opt/cuda-12.8/bin:"))

    def test_camera_binary_and_scene_validation(self) -> None:
        stats = validate_scene(PUBLIC_SCENE)
        self.assertGreater(stats["train_images"], 0)
        self.assertGreater(stats["test_poses"], 0)
        camera = read_first_colmap_camera(PUBLIC_SCENE / "train" / "sparse" / "0" / "cameras.bin")
        self.assertEqual(camera["model"], "SIMPLE_RADIAL")
        self.assertEqual((camera["width"], camera["height"]), (1320, 989))

    def test_prepare_test_colmap(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            adapter, names = prepare_test_colmap(PUBLIC_SCENE, Path(directory) / "adapter")
            manifest = json.loads((adapter / "manifest.json").read_text())
            self.assertEqual(manifest["images"], names)
            self.assertEqual(len(names), validate_scene(PUBLIC_SCENE)["test_poses"])
            self.assertTrue((adapter / "sparse" / "0" / "points3D.bin").is_symlink())
            camera_rows = [
                line
                for line in (adapter / "sparse" / "0" / "cameras.txt").read_text().splitlines()
                if line and not line.startswith("#")
            ]
            self.assertEqual(len(camera_rows), 1, "identical test intrinsics must share one ray cache")
            with Image.open(adapter / "images" / names[0]) as image:
                self.assertEqual(image.size, (1320, 989))

    def test_nested_checkpoint_discovery(self) -> None:
        def checkpoint(path: Path) -> None:
            path.parent.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr("data.pkl", b"checkpoint")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "run-a" / "ours_5000" / "ckpt_5000.pt"
            later = root / "run-b" / "ours_10000" / "ckpt_10000.pt"
            checkpoint(first)
            checkpoint(later)
            self.assertEqual(find_latest_checkpoint(root), later)
            corrupt = root / "run-b" / "ours_15000" / "ckpt_15000.pt"
            corrupt.parent.mkdir(parents=True)
            corrupt.write_bytes(b"partial torch.save")
            self.assertEqual(find_latest_checkpoint(root), later)
            final = root / "run-c" / "ckpt_last.pt"
            checkpoint(final)
            self.assertEqual(find_latest_checkpoint(root), final)

    def test_checkpoint_override_is_sorted_and_includes_final(self) -> None:
        self.assertEqual(
            checkpoint_override([5000, 1000, 5000, 40000], 30000),
            "checkpoint.iterations=[1000,5000,30000]",
        )


if __name__ == "__main__":
    unittest.main()
