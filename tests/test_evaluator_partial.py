from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from src.vtrace.evaluator import read_manifest, validate_submission


def make_scene(root: Path, name: str) -> None:
    test = root / name / "test"
    test.mkdir(parents=True)
    with (test / "test_poses.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image_name", "width", "height"])
        writer.writeheader()
        writer.writerow({"image_name": "frame.JPG", "width": 8, "height": 6})


class PartialEvaluatorTests(unittest.TestCase):
    def test_selected_scene_does_not_require_other_scenes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "data"
            predictions = root / "predictions"
            make_scene(data, "A")
            make_scene(data, "B")
            (predictions / "A").mkdir(parents=True)
            Image.new("RGB", (8, 6)).save(predictions / "A" / "frame.JPG", "JPEG")
            self.assertEqual({entry.scene for entry in read_manifest(data, scenes=["A"])}, {"A"})
            result = validate_submission(data, predictions, scenes=["A"])
            self.assertTrue(result["valid"], result["errors"])
            self.assertEqual(result["expected_images"], 1)


if __name__ == "__main__":
    unittest.main()

