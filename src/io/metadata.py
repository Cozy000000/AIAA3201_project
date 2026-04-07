from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _serialize_path(path: Path, base_dir: Path | None = None) -> str:
    if base_dir is not None:
        try:
            return path.relative_to(base_dir).as_posix()
        except ValueError:
            pass
    return path.as_posix()


def build_metadata(
    *,
    video_name: str,
    fps: float,
    width: int,
    height: int,
    num_frames: int,
    source_video: Path,
    frame_dir: Path,
    sam2_mask_dir: Path,
    baseline_mask_dir: Path,
    base_dir: Path | None = None,
) -> dict[str, Any]:
    """Create the minimum metadata contract for downstream consumption."""
    return {
        "video_name": video_name,
        "fps": fps,
        "width": width,
        "height": height,
        "num_frames": num_frames,
        "source_video": _serialize_path(source_video, base_dir),
        "frame_dir": _serialize_path(frame_dir, base_dir),
        "sam2_mask_dir": _serialize_path(sam2_mask_dir, base_dir),
        "baseline_mask_dir": _serialize_path(baseline_mask_dir, base_dir),
    }


def write_metadata(metadata_path: str | Path, payload: dict[str, Any]) -> Path:
    """Write metadata as pretty-printed JSON."""
    path = Path(metadata_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True)
    return path


def load_metadata(metadata_path: str | Path) -> dict[str, Any]:
    """Load metadata JSON."""
    path = Path(metadata_path)
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)
