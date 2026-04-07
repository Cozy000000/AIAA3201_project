from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from src.io.frame_io import list_frame_paths, read_frame, write_frame

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class VideoInfo:
    fps: float
    width: int
    height: int
    num_frames: int


def inspect_video(video_path: str | Path) -> VideoInfo:
    """Read lightweight metadata from a video file."""
    path = Path(video_path)
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Unable to open video: {path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    num_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    capture.release()

    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid video dimensions for {path}: {width}x{height}")

    return VideoInfo(fps=fps, width=width, height=height, num_frames=num_frames)


def copy_raw_video(input_video: str | Path, raw_video_path: str | Path) -> Path:
    """Copy the source video into the standardized raw directory."""
    source = Path(input_video)
    destination = Path(raw_video_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.resolve() != destination.resolve():
        shutil.copy2(source, destination)
    return destination


def resize_and_pad_frame(
    frame: np.ndarray,
    target_width: int,
    target_height: int,
    keep_aspect: bool = True,
    pad_value: int = 0,
) -> np.ndarray:
    """Resize a frame to a fixed output size, optionally preserving aspect ratio."""
    if frame.ndim != 3:
        raise ValueError("Expected a color frame with shape HxWxC")

    if not keep_aspect:
        return cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)

    source_height, source_width = frame.shape[:2]
    scale = min(target_width / source_width, target_height / source_height)
    resized_width = max(1, int(round(source_width * scale)))
    resized_height = max(1, int(round(source_height * scale)))
    resized = cv2.resize(frame, (resized_width, resized_height), interpolation=cv2.INTER_AREA)

    canvas = np.full(
        (target_height, target_width, frame.shape[2]),
        pad_value,
        dtype=frame.dtype,
    )
    x_offset = (target_width - resized_width) // 2
    y_offset = (target_height - resized_height) // 2
    canvas[y_offset : y_offset + resized_height, x_offset : x_offset + resized_width] = resized
    return canvas


def extract_video_frames(
    input_video: str | Path,
    frame_dir: str | Path,
    target_width: int | None = None,
    target_height: int | None = None,
    keep_aspect: bool = True,
    pad_value: int = 0,
) -> VideoInfo:
    """Extract frames from a video into the canonical PNG sequence."""
    path = Path(input_video)
    output_dir = Path(frame_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Unable to open video: {path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    original_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    original_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    width = target_width or original_width
    height = target_height or original_height

    index = 0
    while True:
        success, frame = capture.read()
        if not success:
            break
        if target_width is not None and target_height is not None:
            frame = resize_and_pad_frame(
                frame,
                target_width=target_width,
                target_height=target_height,
                keep_aspect=keep_aspect,
                pad_value=pad_value,
            )
        write_frame(output_dir, index, frame)
        index += 1

    capture.release()
    LOGGER.info("Extracted %s frames to %s", index, output_dir)

    return VideoInfo(fps=fps, width=width, height=height, num_frames=index)


def compose_video_from_frames(
    frame_dir: str | Path,
    output_video: str | Path,
    fps: float,
    codec: str = "mp4v",
) -> Path:
    """Compose an MP4 video from canonical PNG frames."""
    frames = list_frame_paths(frame_dir)
    if not frames:
        raise FileNotFoundError(f"No PNG frames found in {frame_dir}")

    first_frame = read_frame(frames[0], flags=cv2.IMREAD_COLOR)
    height, width = first_frame.shape[:2]
    output_path = Path(output_video)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*codec),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise IOError(f"Unable to open video writer for {output_path} with codec {codec}")

    for frame_path in frames:
        frame = read_frame(frame_path, flags=cv2.IMREAD_COLOR)
        frame_height, frame_width = frame.shape[:2]
        if frame_width != width or frame_height != height:
            raise ValueError(
                f"Frame size mismatch in {frame_dir}: expected {width}x{height}, "
                f"got {frame_width}x{frame_height} for {frame_path.name}"
            )
        writer.write(frame)

    writer.release()
    LOGGER.info("Composed video %s from %s frames", output_path, len(frames))
    return output_path
