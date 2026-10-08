"""Make a one-frame training sanity set; every split intentionally contains the same training frame."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image


def make_overfit_set(source: Path, output: Path, view: str) -> dict:
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    if (source / "_INCOMPLETE").exists():
        raise ValueError(f"Source export is incomplete: {source}")
    with (source / "frames.csv").open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        fieldnames = reader.fieldnames
        rows = [row for row in reader if row["split"] == "train" and row["view"] == view]
    if not fieldnames or not rows:
        raise ValueError(f"No training frames for {view} in {source}")
    rows.sort(key=lambda row: (row["case_id"], int(row["frame"])))
    chosen = None
    foreground_ratio = None
    for row in rows:
        with Image.open(source / row["mask_png"]) as mask_image:
            binary_mask = mask_image.convert("L")
            mask = np.asarray(binary_mask) > 0
            resized_has_foreground = np.asarray(
                binary_mask.resize((224, 224), Image.Resampling.NEAREST)
            ).any()
        ratio = float(mask.mean())
        if 0.001 <= ratio < 1 and resized_has_foreground:
            chosen, foreground_ratio = row, ratio
            break
    if chosen is None:
        raise ValueError(f"No training frame with at least 0.1% foreground surviving 224px resize for {view}")

    output.mkdir(parents=True)
    marker = output / "_INCOMPLETE"
    marker.write_text("Diagnostic export in progress.\n", encoding="utf-8")
    try:
        with (output / "frames.csv").open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            for split, folder in (("train", "training_set"), ("val", "val_set"), ("test", "test_set")):
                result_row = dict(chosen)
                result_row["split"] = split
                for field, kind in (("image_png", "image"), ("mask_png", "mask")):
                    relative = Path(folder) / kind / Path(chosen[field]).name
                    destination = output / relative
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(source / chosen[field], destination)
                    result_row[field] = relative.as_posix()
                writer.writerow(result_row)
        info = {
            "protocol": "one_training_frame_repeated_across_splits_for_overfit_diagnostic_only",
            "test_is_placeholder_for_official_loader": True,
            "patient_independence_verified": False,
            "source_export": str(source.resolve()),
            "view": view,
            "case_id": chosen["case_id"],
            "frame": int(chosen["frame"]),
            "foreground_ratio": foreground_ratio,
            "case_counts": {"train": 1, "val": 1, "test": 1},
            "frame_counts": {"train": 1, "val": 1, "test": 1},
        }
        (output / "export.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
        marker.unlink()
        return info
    except BaseException:
        # Leave the marker so a partial export cannot be used by the trainer.
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("outputs/usfm_tee_valve"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--view", choices=("LAX", "SAX"), default="LAX")
    args = parser.parse_args()
    print(json.dumps(make_overfit_set(args.source, args.output, args.view), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
