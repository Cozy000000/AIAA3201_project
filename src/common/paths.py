from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class VideoLayout:
    root: Path
    raw_dir: Path
    processed_dir: Path
    frame_dir: Path
    sam2_mask_dir: Path
    baseline_mask_dir: Path
    overlay_dir: Path
    baseline_result_dir: Path
    video_dir: Path
    metadata_path: Path
    raw_video_path: Path


def build_video_layout(output_dir: str | Path) -> VideoLayout:
    """Return the standardized directory layout for one video item."""
    root = Path(output_dir)
    raw_dir = root / "raw"
    processed_dir = root / "processed"
    return VideoLayout(
        root=root,
        raw_dir=raw_dir,
        processed_dir=processed_dir,
        frame_dir=processed_dir / "frames",
        sam2_mask_dir=processed_dir / "masks_sam2",
        baseline_mask_dir=processed_dir / "masks_baseline",
        overlay_dir=processed_dir / "overlays",
        baseline_result_dir=processed_dir / "results_baseline",
        video_dir=processed_dir / "videos",
        metadata_path=processed_dir / "meta.json",
        raw_video_path=raw_dir / "video.mp4",
    )


def ensure_layout(layout: VideoLayout) -> VideoLayout:
    """Create all required directories for the standardized layout."""
    for path in (
        layout.raw_dir,
        layout.processed_dir,
        layout.frame_dir,
        layout.sam2_mask_dir,
        layout.baseline_mask_dir,
        layout.overlay_dir,
        layout.baseline_result_dir,
        layout.video_dir,
    ):
        path.mkdir(parents=True, exist_ok=True)
    return layout


def infer_video_name(output_dir: str | Path, input_video: str | Path | None = None) -> str:
    """Infer a stable video name from output directory or input video path."""
    root = Path(output_dir)
    if root.name:
        return root.name
    if input_video is not None:
        return Path(input_video).stem
    return "video"
