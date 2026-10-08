"""Score official USFM test PNG predictions on the original NIfTI pixel grid."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import nibabel as nib
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from valve_seg.metrics import frame_counts, scores_from_counts, summarize_cases, summarize_frames  # noqa: E402


def evaluate(dataset: Path, predictions: Path, output: Path) -> dict:
    if not predictions.is_dir():
        raise FileNotFoundError(predictions)
    with (dataset / "frames.csv").open(newline="", encoding="utf-8") as file:
        source_rows = [row for row in csv.DictReader(file) if row["split"] == "test"]
    if not source_rows:
        raise ValueError("No test frames in exported dataset")
    by_label: dict[str, np.ndarray] = {}
    frames: list[dict] = []
    for row in source_rows:
        label_path = row["source_label"]
        if label_path not in by_label:
            by_label[label_path] = np.asanyarray(nib.load(label_path).dataobj) > 0
        target = by_label[label_path][:, :, int(row["frame"])]
        predicted_file = predictions / Path(row["mask_png"]).name
        if not predicted_file.is_file():
            raise FileNotFoundError(f"Missing USFM prediction: {predicted_file}")
        with Image.open(predicted_file) as file:
            image = file.convert("L")
            if image.size != (target.shape[1], target.shape[0]):
                image = image.resize((target.shape[1], target.shape[0]), Image.Resampling.NEAREST)
            predicted = np.asarray(image) > 0
        counts = frame_counts(predicted, target)
        frames.append({
            "view": row["view"], "case_id": row["case_id"],
            "frame": int(row["frame"]), **counts, **scores_from_counts(**counts),
        })
    by_case: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in frames:
        by_case[(row["view"], row["case_id"])].append(row)
    cases = [{"view": view, "case_id": case_id, **summarize_frames(rows)}
             for (view, case_id), rows in sorted(by_case.items())]
    summary = {view: summarize_cases([row for row in cases if row["view"] == view])
               for view in sorted({row["view"] for row in cases})}
    output.mkdir(parents=True, exist_ok=False)
    for name, rows in (("frames.csv", frames), ("cases.csv", cases)):
        with (output / name).open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("outputs/usfm_tee_valve"))
    parser.add_argument("--predictions", type=Path, required=True, help="Official mask_pre folder")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(evaluate(args.dataset, args.predictions, args.output), indent=2))


if __name__ == "__main__":
    main()
