from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from src.io.frame_io import list_frame_paths, read_frame
from src.io.video_io import compose_video_from_frames
from src.validation.sanity_checks import assert_matching_dirs


def render_overlay_frame(
    frame: np.ndarray,
    mask: np.ndarray,
    color: tuple[int, int, int] = (0, 0, 255),
    alpha: float = 0.4,
) -> np.ndarray:
    """Overlay a binary mask onto an RGB frame."""
    if mask.ndim == 3:
        mask = np.squeeze(mask)
    if mask.ndim != 2:
        raise ValueError(f"Expected a 2D mask array, got shape {mask.shape}")

    if frame.shape[:2] != mask.shape[:2]:
        raise ValueError(
            f"Frame and mask size mismatch: {frame.shape[:2]} vs {mask.shape[:2]}"
        )

    overlay = frame.copy()
    active = mask > 0
    overlay[active] = (
        (1.0 - alpha) * overlay[active].astype(np.float32)
        + alpha * np.array(color, dtype=np.float32)
    ).astype(np.uint8)
    return overlay


def export_overlay_video(
    frame_dir: str | Path,
    mask_dir: str | Path,
    output_video: str | Path,
    fps: float,
    codec: str = "mp4v",
    color: tuple[int, int, int] = (0, 0, 255),
    alpha: float = 0.4,
) -> Path:
    """Render a temporary overlay sequence and compose it into a video."""
    assert_matching_dirs(frame_dir, mask_dir)

    frame_paths = list_frame_paths(frame_dir)
    mask_paths = list_frame_paths(mask_dir)
    temp_overlay_dir = Path(output_video).parent / f".{Path(output_video).stem}_frames"
    temp_overlay_dir.mkdir(parents=True, exist_ok=True)

    for frame_path, mask_path in zip(frame_paths, mask_paths):
        frame = read_frame(frame_path, flags=cv2.IMREAD_COLOR)
        mask = read_frame(mask_path, flags=cv2.IMREAD_GRAYSCALE)
        overlay = render_overlay_frame(frame, mask, color=color, alpha=alpha)
        output_frame_path = temp_overlay_dir / frame_path.name
        success = cv2.imwrite(str(output_frame_path), overlay)
        if not success:
            raise IOError(f"Failed to write overlay frame: {output_frame_path}")

    return compose_video_from_frames(temp_overlay_dir, output_video, fps=fps, codec=codec)
