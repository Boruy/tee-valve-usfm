"""Read-only audit of the existing TEE NIfTI pairs."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import nibabel as nib
import numpy as np

from valve_seg.data import discover_cases


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "dataset/TEEdataset")
    parser.add_argument("--split", choices=("train", "test", "all"), default="all")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "outputs/data_audit")
    args = parser.parse_args()

    splits = ("train", "test") if args.split == "all" else (args.split,)
    rows = []
    for split in splits:
        for case in discover_cases(args.data_root, split, ("SAX", "LAX")):
            image = nib.load(str(case.image_path))
            label = nib.load(str(case.label_path))
            same_shape = image.shape == label.shape and len(image.shape) == 3
            affine_ok = bool(np.allclose(image.affine, label.affine, atol=1e-4, rtol=0))
            values = np.unique(np.asanyarray(label.dataobj))
            positive_frames = None
            first_positive = None
            if same_shape:
                mask = np.asanyarray(label.dataobj) > 0
                indices = np.flatnonzero(np.any(mask, axis=(0, 1)))
                positive_frames = int(indices.size)
                first_positive = int(indices[0]) if indices.size else None
            rows.append({
                "split": split,
                "view": case.view,
                "case_id": case.case_id,
                "image": str(case.image_path),
                "label": str(case.label_path),
                "shape": "x".join(map(str, image.shape)),
                "frames": image.shape[2] if len(image.shape) == 3 else None,
                "image_dtype": str(image.get_data_dtype()),
                "label_dtype": str(label.get_data_dtype()),
                "label_values": ";".join(map(str, values.tolist())),
                "same_shape": same_shape,
                "affine_ok": affine_ok,
                "positive_frames": positive_frames,
                "empty_frames": (image.shape[2] - positive_frames) if positive_frames is not None else None,
                "first_positive_frame": first_positive,
            })
            print(f"{split} {case.view} {case.case_id}: {image.shape}, affine_ok={affine_ok}")

    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "manifest.csv").open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "protocol_note": "Third NIfTI axis treated as ordered frames; patient-level independence unverified.",
        "cases": len(rows),
        "by_split_view": {
            f"{split}/{view}": {
                "cases": sum(r["split"] == split and r["view"] == view for r in rows),
                "frames": sum(r["frames"] or 0 for r in rows if r["split"] == split and r["view"] == view),
            }
            for split in splits for view in ("SAX", "LAX")
        },
        "shape_mismatches": [r["case_id"] for r in rows if not r["same_shape"]],
        "affine_mismatches": [r["case_id"] for r in rows if not r["affine_ok"]],
        "no_positive_label": [r["case_id"] for r in rows if r["positive_frames"] == 0],
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
