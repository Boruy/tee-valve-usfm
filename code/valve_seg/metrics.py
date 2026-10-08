"""Binary segmentation scores with explicit empty-mask handling."""

from __future__ import annotations

import numpy as np


def frame_counts(prediction: np.ndarray, target: np.ndarray) -> dict[str, int]:
    if prediction.shape != target.shape or prediction.ndim != 2:
        raise ValueError(f"Expected matching 2D masks: {prediction.shape} vs {target.shape}")
    prediction = prediction.astype(bool, copy=False)
    target = target.astype(bool, copy=False)
    return {
        "intersection": int(np.count_nonzero(prediction & target)),
        "predicted_pixels": int(np.count_nonzero(prediction)),
        "target_pixels": int(np.count_nonzero(target)),
    }


def scores_from_counts(intersection: int, predicted_pixels: int, target_pixels: int) -> dict[str, float]:
    total = predicted_pixels + target_pixels
    union = total - intersection
    return {
        "dice": 1.0 if total == 0 else 2.0 * intersection / total,
        "iou": 1.0 if union == 0 else intersection / union,
    }


def summarize_frames(rows: list[dict]) -> dict:
    if not rows:
        raise ValueError("No evaluated frames")
    positives = [row for row in rows if row["target_pixels"] > 0]
    intersection = sum(row["intersection"] for row in rows)
    predicted = sum(row["predicted_pixels"] for row in rows)
    target = sum(row["target_pixels"] for row in rows)
    global_scores = scores_from_counts(intersection, predicted, target)
    return {
        "evaluated_frames": len(rows),
        "positive_target_frames": len(positives),
        "empty_target_frames": len(rows) - len(positives),
        "false_positive_empty_frames": sum(
            row["target_pixels"] == 0 and row["predicted_pixels"] > 0 for row in rows
        ),
        "mean_frame_dice": float(np.mean([row["dice"] for row in rows])),
        "mean_frame_iou": float(np.mean([row["iou"] for row in rows])),
        "mean_positive_frame_dice": (
            float(np.mean([row["dice"] for row in positives])) if positives else None
        ),
        "global_dice": global_scores["dice"],
        "global_iou": global_scores["iou"],
    }


def summarize_cases(rows: list[dict]) -> dict:
    if not rows:
        return {"cases": 0, "evaluated_frames": 0, "macro_case_dice": None, "macro_case_iou": None}
    return {
        "cases": len(rows),
        "evaluated_frames": sum(row["evaluated_frames"] for row in rows),
        "macro_case_dice": float(np.mean([row["mean_frame_dice"] for row in rows])),
        "macro_case_iou": float(np.mean([row["mean_frame_iou"] for row in rows])),
        "macro_case_positive_frame_dice": (
            float(np.mean([row["mean_positive_frame_dice"] for row in rows
                           if row["mean_positive_frame_dice"] is not None]))
            if any(row["mean_positive_frame_dice"] is not None for row in rows) else None
        ),
    }
