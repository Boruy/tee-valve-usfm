"""Image intensity conversion shared by TEE dataset utilities."""

from __future__ import annotations

import numpy as np


def to_uint8(image: np.ndarray) -> np.ndarray:
    """Keep uint8 unchanged; scale other types by finite 0.5/99.5 percentiles."""
    if image.dtype == np.uint8:
        return image
    sample = image.ravel()[:: max(1, image.size // 1_000_000)]
    sample = sample[np.isfinite(sample)]
    if sample.size == 0:
        raise ValueError("Image has no finite pixels")
    low, high = np.percentile(sample, [0.5, 99.5])
    if high <= low:
        return np.zeros(image.shape, dtype=np.uint8)
    return np.clip((np.nan_to_num(image, nan=low, posinf=high, neginf=low) - low)
                   * (255.0 / (high - low)), 0, 255).astype(np.uint8)
