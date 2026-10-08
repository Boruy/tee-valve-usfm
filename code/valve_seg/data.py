"""Pair and load the unmodified TEE NIfTI cine files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np


SUBSETS = {
    ("train", "SAX"): "preprocessed_short_output/short",
    ("test", "SAX"): "preprocessed_short_output_TEST5",
    ("train", "LAX"): "preprocessed_long_output_v2.0/long",
    ("test", "LAX"): "preprocessed_long_test_v2.0/long",
}


@dataclass(frozen=True)
class Case:
    split: str
    view: str
    case_id: str
    image_path: Path
    label_path: Path

    @property
    def output_name(self) -> str:
        return f"{self.view}_{self.case_id}"


def _key(path: Path) -> str:
    name = path.name
    if not name.endswith(".nii.gz"):
        raise ValueError(f"Expected .nii.gz: {path}")
    name = name[:-7]
    while name.endswith("_img") or name.endswith("_label"):
        name = name.rsplit("_", 1)[0]
    return name


def discover_cases(data_root: Path, split: str, views: tuple[str, ...]) -> list[Case]:
    """Discover image/label pairs, including the known *_img_label filename."""
    records: list[Case] = []
    for view in views:
        subset = data_root / SUBSETS[(split, view)]
        if not subset.is_dir():
            raise FileNotFoundError(f"Dataset subset missing: {subset}")
        by_directory: dict[Path, dict[str, dict[str, Path]]] = {}
        for path in sorted(subset.rglob("*.nii.gz")):
            kind = (
                "label" if path.name.endswith("_label.nii.gz")
                else "image" if path.name.endswith("_img.nii.gz")
                else None
            )
            if kind is None:
                continue
            slot = by_directory.setdefault(path.parent, {}).setdefault(_key(path), {})
            if kind in slot:
                raise ValueError(f"Duplicate {kind} in {path.parent}: {_key(path)}")
            slot[kind] = path
        for directory, pairs in sorted(by_directory.items()):
            for key, pair in sorted(pairs.items()):
                if set(pair) != {"image", "label"}:
                    raise ValueError(f"Incomplete NIfTI pair in {directory}: {key}, {set(pair)}")
                relative = directory.relative_to(subset)
                parts = [p for p in relative.parts if p != "."] + [key]
                case_id = "_".join(parts)
                records.append(Case(split, view, case_id, pair["image"], pair["label"]))
    return sorted(records, key=lambda c: (c.view, c.case_id))


def load_case(case: Case) -> tuple[nib.Nifti1Image, np.ndarray, np.ndarray]:
    """Return image and binary label arrays in native H x W x T index order."""
    image_nii = nib.load(str(case.image_path))
    label_nii = nib.load(str(case.label_path))
    if len(image_nii.shape) != 3 or image_nii.shape != label_nii.shape:
        raise ValueError(
            f"Expected equal 3D HxWxT shapes for {case.output_name}: "
            f"{image_nii.shape} vs {label_nii.shape}"
        )
    if not np.allclose(image_nii.affine, label_nii.affine, rtol=0, atol=1e-4):
        raise ValueError(f"Image/label affine mismatch: {case.output_name}")
    image = np.asanyarray(image_nii.dataobj)
    label = np.asanyarray(label_nii.dataobj) > 0
    if image.shape[2] < 2:
        raise ValueError(f"Need at least two frames: {case.output_name}")
    return image_nii, image, label
