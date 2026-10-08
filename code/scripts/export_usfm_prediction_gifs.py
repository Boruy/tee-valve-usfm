"""Render official USFM test PNG predictions as one comparison GIF per cine sequence."""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from valve_seg.metrics import frame_counts, scores_from_counts  # noqa: E402


def _read_gray(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("L")


def _overlay(gray: np.ndarray, mask: np.ndarray, color: tuple[int, int, int]) -> Image.Image:
    rgb = np.repeat(gray[:, :, None], 3, axis=2)
    rgb[mask] = (0.4 * rgb[mask] + 0.6 * np.asarray(color)).astype(np.uint8)
    return Image.fromarray(rgb)


def _render(row: dict[str, str], dataset: Path, predictions: Path, panel_width: int) -> Image.Image:
    image = _read_gray(dataset / row["image_png"])
    target_image = _read_gray(dataset / row["mask_png"])
    prediction_image = _read_gray(predictions / Path(row["mask_png"]).name)
    if target_image.size != image.size:
        raise ValueError(f"Image/label size mismatch: {row['image_png']}")
    if prediction_image.size != image.size:
        prediction_image = prediction_image.resize(image.size, Image.Resampling.NEAREST)
    gray = np.asarray(image)
    target = np.asarray(target_image) > 0
    predicted = np.asarray(prediction_image) > 0
    dice = scores_from_counts(**frame_counts(predicted, target))["dice"]

    panel_height = max(1, round(gray.shape[0] * panel_width / gray.shape[1]))
    canvas = Image.new("RGB", (3 * panel_width, panel_height + 48), "black")
    draw = ImageDraw.Draw(canvas)
    draw.text((6, 4), f"{row['view']} {row['case_id']} | Frame {row['frame']} | Dice {dice:.3f}", fill="white")
    panels = (
        (Image.fromarray(gray).convert("RGB"), "Ultrasound"),
        (_overlay(gray, target, (0, 255, 0)), "Ground truth"),
        (_overlay(gray, predicted, (255, 0, 0)), "USFM prediction"),
    )
    for index, (panel, label) in enumerate(panels):
        x = index * panel_width
        draw.text((x + 6, 26), label, fill="white")
        canvas.paste(panel.resize((panel_width, panel_height), Image.Resampling.BILINEAR), (x, 48))
    return canvas


def export_gifs(
    dataset: Path,
    predictions: Path,
    output: Path,
    fps: int = 8,
    stride: int = 1,
    panel_width: int = 256,
    view: str = "both",
    cases: list[str] | None = None,
) -> list[Path]:
    if fps < 1 or stride < 1 or panel_width < 64:
        raise ValueError("fps and stride must be positive; panel-width must be at least 64")
    if not predictions.is_dir():
        raise FileNotFoundError(f"USFM mask_pre directory missing: {predictions}")
    if output.exists():
        raise FileExistsError(f"GIF output already exists: {output}")
    with (dataset / "frames.csv").open(newline="", encoding="utf-8-sig") as file:
        rows = [row for row in csv.DictReader(file) if row["split"] == "test"]
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if view != "both" and row["view"] != view:
            continue
        key = (row["view"], row["case_id"])
        if cases and row["case_id"] not in cases and f"{key[0]}_{key[1]}" not in cases:
            continue
        grouped[key].append(row)
    if not grouped:
        raise ValueError("No matching test sequences in frames.csv")
    if cases:
        found = {case for view_name, case in grouped} | {f"{view_name}_{case}" for view_name, case in grouped}
        missing = set(cases) - found
        if missing:
            raise ValueError(f"Unknown test sequence(s): {sorted(missing)}")

    # Check the complete prediction first, even if --stride will display fewer frames.
    for key, sequence in grouped.items():
        sequence.sort(key=lambda row: int(row["frame"]))
        indices = [int(row["frame"]) for row in sequence]
        if indices != list(range(len(indices))):
            raise ValueError(f"Missing or duplicate source frame in {key}: {indices}")
        for row in sequence:
            for path in (
                dataset / row["image_png"],
                dataset / row["mask_png"],
                predictions / Path(row["mask_png"]).name,
            ):
                if not path.is_file():
                    raise FileNotFoundError(f"Missing GIF input: {path}")

    output.mkdir(parents=True)
    saved: list[Path] = []
    for (view_name, case_id), sequence in sorted(grouped.items()):
        frames = [_render(row, dataset, predictions, panel_width) for row in sequence[::stride]]
        path = output / f"{view_name}_{case_id}.gif"
        try:
            frames[0].save(
                path,
                save_all=True,
                append_images=frames[1:],
                duration=round(1000 / fps),
                loop=0,
                optimize=True,
                disposal=2,
            )
        finally:
            for frame in frames:
                frame.close()
        saved.append(path)
        print(f"{path} ({len(sequence[::stride])} frames, {path.stat().st_size / 1048576:.1f} MiB)", flush=True)
    return saved


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("outputs/usfm_tee_valve"))
    parser.add_argument("--predictions", type=Path, required=True, help="Official USFM best_test_dice*/mask_pre directory")
    parser.add_argument("--output", type=Path, required=True, help="New directory for one GIF per test sequence")
    parser.add_argument("--case", action="append", help="Case ID or VIEW_case ID; repeatable")
    parser.add_argument("--view", choices=("SAX", "LAX", "both"), default="both")
    parser.add_argument("--fps", type=int, default=8, help="Playback speed, not ultrasound acquisition speed")
    parser.add_argument("--stride", type=int, default=1, help="Display every Nth frame; all predictions are checked")
    parser.add_argument("--panel-width", type=int, default=256)
    args = parser.parse_args()
    saved = export_gifs(
        args.dataset, args.predictions, args.output, args.fps, args.stride,
        args.panel_width, args.view, args.case,
    )
    print(f"Exported {len(saved)} GIF(s) to {args.output}")


if __name__ == "__main__":
    main()
