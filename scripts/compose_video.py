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
from src.io.metadata import load_metadata
from src.io.video_io import compose_video_from_frames
from src.validation.sanity_checks import summarize_image_dir

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compose an MP4 video from extracted frames.")
    parser.add_argument("--input_video", type=Path, default=None)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--save_overlay", action="store_true")
    parser.add_argument("--frames_dir", type=Path, default=None)
    parser.add_argument("--output_video", type=Path, default=None)
    parser.add_argument("--fps", type=float, default=None)
    parser.add_argument("--codec", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    setup_logging(get_nested(config, "logging.level", "INFO"))

    layout = ensure_layout(build_video_layout(args.output_dir))
    frames_dir = args.frames_dir or layout.frame_dir

    if args.fps is not None:
        fps = args.fps
    elif layout.metadata_path.exists():
        fps = float(load_metadata(layout.metadata_path)["fps"])
    else:
        fps = float(get_nested(config, "video.fps", 25.0))

    codec = args.codec or str(get_nested(config, "video.codec", "mp4v"))
    output_video = args.output_video or (layout.video_dir / "reconstructed.mp4")

    summary = summarize_image_dir(frames_dir)
    compose_video_from_frames(frames_dir, output_video, fps=fps, codec=codec)

    LOGGER.info("Frames directory: %s", frames_dir)
    LOGGER.info("Output video: %s", output_video)
    LOGGER.info(
        "Composed %s frames at %sx%s and %.3f FPS",
        summary.frame_count,
        summary.frame_size[0],
        summary.frame_size[1],
        fps,
    )


if __name__ == "__main__":
    main()
