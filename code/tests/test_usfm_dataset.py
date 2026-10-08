"""Check that USFM export preserves masks and refuses patient leakage."""

from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from _output_dir import OUTPUTS_ROOT, replace_existing_output  # noqa: E402
from prepare_usfm_dataset import export  # noqa: E402
from eval_usfm_predictions import evaluate  # noqa: E402


class TestUSFMExport(unittest.TestCase):
    def _case(self, root: Path, folder: str, name: str) -> Path:
        location = root / folder / name
        location.mkdir(parents=True)
        image = np.zeros((8, 7, 2), dtype=np.uint8)
        image[2:5, 1:4, 1] = 180
        label = np.zeros((8, 7, 2), dtype=np.uint16)
        label[2:5, 1:4, 1] = 65535
        source = location / f"{name}_img.nii.gz"
        nib.save(nib.Nifti1Image(image, np.eye(4)), source)
        nib.save(nib.Nifti1Image(label, np.eye(4)), location / f"{name}_label.nii.gz")
        return source

    def test_export_preserves_frame_order_and_binary_mask(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "data"
            self._case(root, "preprocessed_short_output/short", "train1")
            self._case(root, "preprocessed_short_output/short", "train2")
            self._case(root, "preprocessed_short_output_TEST5", "test1")
            output = Path(temporary) / "out"
            info = export(root, output, views=("SAX",), seed=3)
            self.assertEqual(info["frame_counts"], {"train": 2, "val": 2, "test": 2})
            with (output / "frames.csv").open(newline="", encoding="utf-8") as file:
                rows = list(csv.DictReader(file))
            self.assertEqual(len({row["image_png"] for row in rows}), 6)
            test_rows = sorted((r for r in rows if r["split"] == "test"), key=lambda r: int(r["frame"]))
            masks = [np.asarray(Image.open(output / row["mask_png"])) for row in test_rows]
            self.assertEqual(int(masks[0].sum()), 0)
            self.assertEqual(int(masks[1].sum()), 9 * 255)
            self.assertEqual(set(np.unique(masks[1])), {0, 255})
            predictions = Path(temporary) / "mask_pre"
            predictions.mkdir()
            for row in test_rows:
                image = np.asarray(Image.open(output / row["mask_png"]))
                Image.fromarray((image > 0).astype(np.uint8)).save(predictions / Path(row["mask_png"]).name)
            summary = evaluate(output, predictions, Path(temporary) / "scores")
            self.assertEqual(summary["SAX"]["macro_case_dice"], 1.0)

    def test_rejects_same_patient_in_multiple_splits(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "data"
            train = self._case(root, "preprocessed_short_output/short", "train1")
            test = self._case(root, "preprocessed_short_output_TEST5", "test1")
            manifest = Path(temporary) / "patients.csv"
            with manifest.open("w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=["anonymous_patient_id", "source_image", "split", "review_status"])
                writer.writeheader()
                for source, split in ((train, "train"), (test, "test")):
                    writer.writerow({"anonymous_patient_id": "person1", "source_image": str(source), "split": split, "review_status": "approved"})
            with self.assertRaisesRegex(ValueError, "multiple splits"):
                export(root, Path(temporary) / "out", views=("SAX",), manifest=manifest)

    def test_exploratory_export_records_invalid_affine(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "data"
            bad = self._case(root, "preprocessed_short_output/short", "bad")
            self._case(root, "preprocessed_short_output/short", "good")
            self._case(root, "preprocessed_short_output_TEST5", "test")
            label_path = bad.with_name("bad_label.nii.gz")
            label = nib.load(label_path)
            shifted = np.eye(4)
            shifted[0, 3] = 3
            nib.save(nib.Nifti1Image(np.asanyarray(label.dataobj), shifted), label_path)
            info = export(root, Path(temporary) / "out", views=("SAX",), skip_invalid=True)
            self.assertEqual(len(info["skipped_cases"]), 1)
            self.assertIn("affine mismatch", info["skipped_cases"][0]["reason"])
            self.assertEqual(sum(info["case_counts"].values()), 2)

    def test_rerun_replaces_incomplete_export(self):
        OUTPUTS_ROOT.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=OUTPUTS_ROOT) as temporary:
            root = Path(temporary) / "data"
            self._case(root, "preprocessed_short_output/short", "train1")
            self._case(root, "preprocessed_short_output/short", "train2")
            self._case(root, "preprocessed_short_output_TEST5", "test1")
            output = Path(temporary) / "out"
            output.mkdir()
            (output / "_INCOMPLETE").write_text("stale", encoding="utf-8")
            (output / "stale.png").write_bytes(b"stale")

            info = export(root, output, views=("SAX",), seed=3)

            self.assertEqual(info["frame_counts"], {"train": 2, "val": 2, "test": 2})
            self.assertTrue((output / "export.json").is_file())
            self.assertFalse((output / "_INCOMPLETE").exists())
            self.assertFalse((output / "stale.png").exists())

    def test_overwrite_refuses_input_and_paths_outside_outputs(self):
        OUTPUTS_ROOT.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=OUTPUTS_ROOT) as temporary:
            output = Path(temporary) / "out"
            output.mkdir()
            input_file = output / "checkpoint.pth"
            input_file.touch()
            with self.assertRaisesRegex(ValueError, "Refusing to overwrite input"):
                replace_existing_output(output, protected=(input_file,))
            self.assertTrue(input_file.exists())
        with tempfile.TemporaryDirectory() as temporary:
            outside = Path(temporary) / "out"
            outside.mkdir()
            with self.assertRaisesRegex(ValueError, "outside"):
                replace_existing_output(outside)
            self.assertTrue(outside.exists())


if __name__ == "__main__":
    unittest.main()
