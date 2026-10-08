"""Safely replace generated directories inside this project's outputs folder."""

from __future__ import annotations

import shutil
from pathlib import Path


OUTPUTS_ROOT = (Path(__file__).resolve().parents[2] / "outputs").resolve()


def replace_existing_output(output: Path, *, protected: tuple[Path, ...] = ()) -> bool:
    """Remove a previous run's directory, never an input or a path outside outputs/."""
    if not output.exists() and not output.is_symlink():
        return False

    target = output.resolve()
    if target == OUTPUTS_ROOT or OUTPUTS_ROOT not in target.parents:
        raise ValueError(f"Refusing to overwrite outside {OUTPUTS_ROOT}: {output}")
    if output.is_symlink() or getattr(output, "is_junction", lambda: False)() or not output.is_dir():
        raise ValueError(f"Refusing to overwrite a link or non-directory: {output}")
    for source in protected:
        source_path = source.resolve()
        if source_path == target or target in source_path.parents or source_path in target.parents:
            raise ValueError(f"Refusing to overwrite input {source}: {output}")

    shutil.rmtree(output)
    return True
