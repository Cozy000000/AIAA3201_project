from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.config import get_nested, load_config
from src.common.logging_utils import setup_logging
from src.common.paths import build_video_layout, ensure_layout
from src.io.video_io import inspect_video, resize_and_pad_frame

import cv2

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Normalize an input video into the standardized raw/video.mp4 slot.")
    parser.add_argument("--input_video", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--save_overlay", action="store_true")
    parser.add_argument("--target_width", type=int, default=None)
    parser.add_argument("--target_height", type=int, default=None)
    parser.add_argument("--fps", type=float, default=None)
    parser.add_argument("--keep_aspect", action="store_true")
    parser.add_argument("--pad_value", type=int, default=None)
    parser.add_argument("--codec", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    setup_logging(get_nested(config, "logging.level", "INFO"))

    layout = ensure_layout(build_video_layout(args.output_dir))

    source_info = inspect_video(args.input_video)
    target_width = args.target_width or get_nested(config, "video.target_width") or source_info.width
    target_height = args.target_height or get_nested(config, "video.target_height") or source_info.height
    target_fps = float(args.fps or get_nested(config, "video.fps") or source_info.fps or 25.0)
    keep_aspect = args.keep_aspect or bool(get_nested(config, "video.keep_aspect", False))
    pad_value = (
        args.pad_value
        if args.pad_value is not None
        else int(get_nested(config, "video.pad_value", 0))
    )
    codec = args.codec or str(get_nested(config, "video.codec", "mp4v"))

    capture = cv2.VideoCapture(str(args.input_video))
    if not capture.isOpened():
        raise FileNotFoundError(f"Unable to open video: {args.input_video}")

    writer = cv2.VideoWriter(
        str(layout.raw_video_path),
        cv2.VideoWriter_fourcc(*codec),
        target_fps,
        (int(target_width), int(target_height)),
    )
    if not writer.isOpened():
        raise IOError(f"Unable to open video writer: {layout.raw_video_path}")

    frame_count = 0
    while True:
        success, frame = capture.read()
        if not success:
            break
        normalized = resize_and_pad_frame(
            frame,
            target_width=int(target_width),
            target_height=int(target_height),
            keep_aspect=keep_aspect,
            pad_value=pad_value,
        )
        writer.write(normalized)
        frame_count += 1

    capture.release()
    writer.release()

    LOGGER.info("Prepared video: %s", layout.raw_video_path)
    LOGGER.info("Input video: %s", args.input_video)
    LOGGER.info(
        "Normalized to %sx%s at %.3f FPS with %s frames",
        target_width,
        target_height,
        target_fps,
        frame_count,
    )


if __name__ == "__main__":
    main()
