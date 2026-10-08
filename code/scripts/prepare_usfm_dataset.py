"""Export TEE NIfTI cine frames to the official USFM SegBase PNG layout."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from _output_dir import replace_existing_output

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from valve_seg.data import Case, discover_cases, load_case  # noqa: E402
from valve_seg.image import to_uint8  # noqa: E402


def _all_cases(root: Path, views: tuple[str, ...]) -> list[Case]:
    return [
        *discover_cases(root, "train", views),
        *discover_cases(root, "test", views),
    ]


def _folder_assignment(cases: list[Case], val_fraction: float, seed: int) -> dict[Path, str]:
    """Exploratory split: hold out complete training sequences within each view."""
    assignment = {c.image_path.resolve(): "test" for c in cases if c.split == "test"}
    by_view: dict[str, list[Case]] = defaultdict(list)
    for case in cases:
        if case.split == "train":
            by_view[case.view].append(case)
    for view, group in by_view.items():
        if len(group) < 2:
            raise ValueError(f"Need at least two training sequences in {view} for train/val")
        ranked = sorted(
            group,
            key=lambda c: hashlib.sha256(f"{seed}:{c.view}:{c.case_id}".encode()).hexdigest(),
        )
        n_val = min(len(group) - 1, max(1, round(len(group) * val_fraction)))
        for i, case in enumerate(ranked):
            assignment[case.image_path.resolve()] = "val" if i < n_val else "train"
    return assignment


def _manifest_assignment(cases: list[Case], manifest: Path, root: Path) -> dict[Path, str]:
    """Use a reviewed, deidentified patient split; require every discovered case."""
    expected = {c.image_path.resolve() for c in cases}
    assignment: dict[Path, str] = {}
    patient_splits: dict[str, str] = {}
    with manifest.open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        required = {"anonymous_patient_id", "source_image", "split", "review_status"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"Manifest needs columns: {', '.join(sorted(required))}")
        for row in reader:
            source = Path(row["source_image"])
            source = (source if source.is_absolute() else root / source).resolve()
            if source not in expected:
                raise ValueError(f"Manifest image is outside discovered main dataset: {source}")
            if source in assignment:
                raise ValueError(f"Duplicate manifest image: {source}")
            split = row["split"].strip().lower()
            patient = row["anonymous_patient_id"].strip()
            if split not in {"train", "val", "test"} or not patient:
                raise ValueError(f"Invalid patient or split for {source}")
            if row["review_status"].strip().lower() != "approved":
                raise ValueError(f"Manifest row must have review_status=approved: {source}")
            if patient in patient_splits and patient_splits[patient] != split:
                raise ValueError(f"Patient {patient} occurs in multiple splits")
            patient_splits[patient] = split
            assignment[source] = split
    if set(assignment) != expected:
        raise ValueError(f"Manifest misses {len(expected - set(assignment))} discovered images")
    if set(assignment.values()) != {"train", "val", "test"}:
        raise ValueError("Manifest must contain train, val, and test cases")
    return assignment


def export(
    data_root: Path,
    output: Path,
    views: tuple[str, ...] = ("SAX", "LAX"),
    manifest: Path | None = None,
    val_fraction: float = 0.15,
    seed: int = 42,
    max_cases: int | None = None,
    skip_invalid: bool = False,
) -> dict:
    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction must be between 0 and 1")
    cases = _all_cases(data_root, views)
    if manifest and skip_invalid:
        raise ValueError("Cannot skip invalid cases with an approved patient manifest")
    assignment = (
        _manifest_assignment(cases, manifest, data_root)
        if manifest else _folder_assignment(cases, val_fraction, seed)
    )
    if max_cases is not None:
        if max_cases < 1:
            raise ValueError("max_cases must be positive")
        cases = cases[:max_cases]
    protected = (data_root,) + ((manifest,) if manifest else ())
    if replace_existing_output(output, protected=protected):
        print(f"Overwriting previous USFM dataset export: {output}", flush=True)
    staging = output
    stage_names = {"train": "training_set", "val": "val_set", "test": "test_set"}
    counts: Counter[str] = Counter()
    case_counts: Counter[str] = Counter()
    skipped: list[dict[str, str]] = []
    staging.mkdir(parents=True)
    marker = staging / "_INCOMPLETE"
    marker.write_text("Export is in progress; do not use this directory for training.\n", encoding="utf-8")
    try:
        with (staging / "frames.csv").open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(
                file,
                fieldnames=["split", "view", "case_id", "frame", "image_png", "mask_png", "source_image", "source_label"],
            )
            writer.writeheader()
            for case in cases:
                split = assignment[case.image_path.resolve()]
                try:
                    _, image, label = load_case(case)
                except ValueError as error:
                    if not skip_invalid:
                        raise
                    skipped.append({"view": case.view, "case_id": case.case_id, "reason": str(error)})
                    continue
                case_counts[split] += 1
                # Preserve original uint8 values; scale other image types per sequence.
                image_u8 = to_uint8(image)
                source_relative = case.image_path.resolve().relative_to(data_root.resolve())
                token = hashlib.sha256(source_relative.as_posix().encode("utf-8")).hexdigest()[:12]
                prefix = f"{case.view}_{token}"
                for frame in range(image.shape[2]):
                    name = f"{prefix}_t{frame:04d}.png"
                    image_rel = Path(stage_names[split]) / "image" / name
                    mask_rel = Path(stage_names[split]) / "mask" / name
                    (staging / image_rel).parent.mkdir(parents=True, exist_ok=True)
                    (staging / mask_rel).parent.mkdir(parents=True, exist_ok=True)
                    Image.fromarray(np.ascontiguousarray(image_u8[:, :, frame])).save(staging / image_rel)
                    Image.fromarray(np.ascontiguousarray(label[:, :, frame].astype(np.uint8) * 255)).save(staging / mask_rel)
                    writer.writerow({
                        "split": split, "view": case.view, "case_id": case.case_id,
                        "frame": frame, "image_png": image_rel.as_posix(),
                        "mask_png": mask_rel.as_posix(), "source_image": str(case.image_path),
                        "source_label": str(case.label_path),
                    })
                    counts[split] += 1
        info = {
            "protocol": "patient_manifest" if manifest else "exploratory_folder_split",
            "patient_independence_verified": manifest is not None,
            "source": str(data_root.resolve()),
            "manifest": str(manifest.resolve()) if manifest else None,
            "seed": seed if not manifest else None,
            "val_fraction": val_fraction if not manifest else None,
            "case_counts": dict(case_counts),
            "frame_counts": dict(counts),
            "image_policy": "uint8 unchanged; other dtypes per-sequence 0.5/99.5 percentile to uint8",
            "mask_values": [0, 255],
            "skipped_cases": skipped,
        }
        (staging / "export.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
        marker.unlink()
        return info
    except BaseException:
        # Keep the marker and partial files for inspection; rerunning replaces them.
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("dataset/TEEdataset"))
    parser.add_argument("--output", type=Path, default=Path("outputs/usfm_tee_valve"))
    parser.add_argument("--views", nargs="+", choices=["SAX", "LAX"], default=["SAX", "LAX"])
    parser.add_argument("--patient-manifest", type=Path)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-cases", type=int, help="Conversion smoke test only")
    parser.add_argument("--skip-invalid", action="store_true", help="Exploratory only; record invalid cases")
    args = parser.parse_args()
    info = export(args.data_root, args.output, tuple(args.views), args.patient_manifest,
                  args.val_fraction, args.seed, args.max_cases, args.skip_invalid)
    print(json.dumps(info, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
