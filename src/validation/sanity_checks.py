from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.io.frame_io import assert_expected_filenames, list_frame_paths, read_frame


@dataclass(frozen=True)
class AlignmentSummary:
    frame_count: int
    frame_size: tuple[int, int]


def summarize_image_dir(image_dir: str | Path) -> AlignmentSummary:
    """Return frame count and canonical size for a PNG directory."""
    paths = list_frame_paths(image_dir)
    if not paths:
        raise FileNotFoundError(f"No PNG files found in {image_dir}")

    assert_expected_filenames(paths)
    first = read_frame(paths[0])
    height, width = first.shape[:2]

    for path in paths[1:]:
        image = read_frame(path)
        if image.shape[:2] != (height, width):
            raise ValueError(
                f"Size mismatch in {image_dir}: expected {(height, width)}, got {image.shape[:2]} "
                f"for {path.name}"
            )

    return AlignmentSummary(frame_count=len(paths), frame_size=(width, height))


def assert_matching_dirs(reference_dir: str | Path, target_dir: str | Path) -> None:
    """Check naming, count, and size equality between two PNG directories."""
    reference_paths = list_frame_paths(reference_dir)
    target_paths = list_frame_paths(target_dir)

    if not reference_paths:
        raise FileNotFoundError(f"Reference directory is empty: {reference_dir}")
    if not target_paths:
        raise FileNotFoundError(f"Target directory is empty: {target_dir}")

    assert_expected_filenames(reference_paths)
    assert_expected_filenames(target_paths)

    if len(reference_paths) != len(target_paths):
        raise ValueError(
            f"Frame count mismatch: {reference_dir} has {len(reference_paths)}, "
            f"{target_dir} has {len(target_paths)}"
        )

    for reference_path, target_path in zip(reference_paths, target_paths):
        if reference_path.name != target_path.name:
            raise ValueError(
                f"Filename mismatch: {reference_path.name} vs {target_path.name}"
            )

        reference = read_frame(reference_path)
        target = read_frame(target_path)
        if reference.shape[:2] != target.shape[:2]:
            raise ValueError(
                f"Resolution mismatch for {reference_path.name}: "
                f"{reference.shape[1]}x{reference.shape[0]} vs "
                f"{target.shape[1]}x{target.shape[0]}"
            )


def assert_mask_binary(mask_dir: str | Path) -> None:
    """Check that every mask image uses only 0 and 255 values."""
    for path in list_frame_paths(mask_dir):
        image = read_frame(path)
        unique_values = set(int(value) for value in image.flatten())
        if not unique_values.issubset({0, 255}):
            raise ValueError(f"Mask {path} is not binary 0/255: {sorted(unique_values)}")
