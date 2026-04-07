from __future__ import annotations

from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

FRAME_NAME_WIDTH = 5


def frame_name(index: int) -> str:
    """Return the canonical zero-padded frame filename."""
    return f"{index:0{FRAME_NAME_WIDTH}d}.png"


def write_frame(frame_dir: str | Path, index: int, frame: np.ndarray) -> Path:
    """Write one frame using canonical naming."""
    output_path = Path(frame_dir) / frame_name(index)
    success = cv2.imwrite(str(output_path), frame)
    if not success:
        raise IOError(f"Failed to write frame to {output_path}")
    return output_path


def list_frame_paths(frame_dir: str | Path) -> list[Path]:
    """List PNG frames in sorted order."""
    return sorted(Path(frame_dir).glob("*.png"))


def read_frame(path: str | Path, flags: int = cv2.IMREAD_UNCHANGED) -> np.ndarray:
    """Read a frame image and raise a clear error if loading fails."""
    image = cv2.imread(str(path), flags)
    if image is None:
        raise FileNotFoundError(f"Unable to read image: {path}")
    return image


def assert_expected_filenames(paths: Iterable[Path]) -> None:
    """Validate canonical frame numbering without gaps relative to order."""
    for index, path in enumerate(paths):
        expected = frame_name(index)
        if path.name != expected:
            raise ValueError(
                f"Unexpected filename order: expected {expected}, got {path.name}"
            )
