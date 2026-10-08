"""Create a small, sequence-disjoint USFM pilot from an exported dataset."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path


LIMITS = {"train": 4, "val": 1, "test": 1}


def make_pilot(source: Path, output: Path, views: tuple[str, ...], max_frames: int) -> dict:
    if output.exists():
        raise FileExistsError(output)
    if max_frames < 1:
        raise ValueError("max-frames must be positive")
    info = json.loads((source / "export.json").read_text(encoding="utf-8"))
    if (source / "_INCOMPLETE").exists():
        raise ValueError("Source export is incomplete")
    with (source / "frames.csv").open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    by_case: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for row in rows:
        if row["view"] in views:
            by_case[(row["split"], row["view"], row["case_id"])].append(row)
    selected: list[dict] = []
    counts = Counter()
    for split in ("train", "val", "test"):
        for view in views:
            keys = sorted(key for key in by_case if key[0] == split and key[1] == view)
            if len(keys) < LIMITS[split]:
                raise ValueError(f"Not enough {split} {view} sequences")
            for key in keys[: LIMITS[split]]:
                frames = sorted(by_case[key], key=lambda row: int(row["frame"]))
                # Evenly sample cine phase rather than taking only the beginning.
                indices = [round(i * (len(frames) - 1) / (min(max_frames, len(frames)) - 1))
                           for i in range(min(max_frames, len(frames)))] if len(frames) > 1 and max_frames > 1 else [0]
                selected.extend(frames[i] for i in indices)
                counts[f"{split}_{view}_cases"] += 1
                counts[f"{split}_{view}_frames"] += len(indices)
    output.mkdir(parents=True)
    marker = output / "_INCOMPLETE"
    marker.write_text("Pilot preparation in progress.\n", encoding="utf-8")
    with (output / "frames.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in selected:
            for field in ("image_png", "mask_png"):
                source_file = source / row[field]
                destination = output / row[field]
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source_file, destination)
            writer.writerow(row)
    result = {
        "protocol": "pilot_subset_of_" + info["protocol"],
        "patient_independence_verified": info["patient_independence_verified"],
        "source_export": str(source.resolve()),
        "max_frames_per_case": max_frames,
        "counts": dict(counts),
    }
    (output / "export.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    marker.unlink()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("outputs/usfm_tee_valve"))
    parser.add_argument("--output", type=Path, default=Path("outputs/usfm_pilot"))
    parser.add_argument("--views", nargs="+", choices=["SAX", "LAX"], default=["SAX", "LAX"])
    parser.add_argument("--max-frames", type=int, default=16)
    args = parser.parse_args()
    print(json.dumps(make_pilot(args.source, args.output, tuple(args.views), args.max_frames), indent=2))


if __name__ == "__main__":
    main()
