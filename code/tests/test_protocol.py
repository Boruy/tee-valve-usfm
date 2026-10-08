"""Checks for data and scoring mistakes that would invalidate a baseline."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import nibabel as nib
import numpy as np

from valve_seg.data import discover_cases, load_case
from valve_seg.metrics import frame_counts, scores_from_counts


class TestDataProtocol(unittest.TestCase):
    def test_anomalous_label_name_pairs_and_mask_loading(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary) / "preprocessed_short_output_TEST5/test01"
            folder.mkdir(parents=True)
            image = np.zeros((8, 6, 3), dtype=np.uint8)
            label = np.zeros_like(image, dtype=np.uint16)
            label[2:4, 1:3, 1] = 65535
            nib.save(nib.Nifti1Image(image, np.eye(4)), folder / "case_img.nii.gz")
            nib.save(nib.Nifti1Image(label, np.eye(4)), folder / "case_img_label.nii.gz")
            cases = discover_cases(Path(temporary), "test", ("SAX",))
            self.assertEqual(len(cases), 1)
            _, _, mask = load_case(cases[0])
            self.assertEqual(mask.shape, (8, 6, 3))
            self.assertEqual(int(mask.sum()), 4)

    def test_affine_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary) / "preprocessed_short_output_TEST5/test01"
            folder.mkdir(parents=True)
            data = np.zeros((8, 6, 3), dtype=np.uint8)
            shifted = np.eye(4)
            shifted[0, 3] = 2
            nib.save(nib.Nifti1Image(data, np.eye(4)), folder / "case_img.nii.gz")
            nib.save(nib.Nifti1Image(data, shifted), folder / "case_label.nii.gz")
            case = discover_cases(Path(temporary), "test", ("SAX",))[0]
            with self.assertRaisesRegex(ValueError, "affine mismatch"):
                load_case(case)

    def test_empty_and_nonempty_mask_scores(self):
        empty = np.zeros((3, 3), dtype=bool)
        counts = frame_counts(empty, empty)
        self.assertEqual(scores_from_counts(**counts), {"dice": 1.0, "iou": 1.0})
        positive = empty.copy()
        positive[0, 0] = True
        counts = frame_counts(empty, positive)
        self.assertEqual(scores_from_counts(**counts), {"dice": 0.0, "iou": 0.0})


if __name__ == "__main__":
    unittest.main()
