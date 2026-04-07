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
from src.common.paths import build_video_layout, ensure_layout, infer_video_name
from src.io.metadata import build_metadata, write_metadata
from src.io.video_io import copy_raw_video, extract_video_frames

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract video frames into the standardized layout.")
    parser.add_argument("--input_video", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--save_overlay", action="store_true")
    parser.add_argument("--target_width", type=int, default=None)
    parser.add_argument("--target_height", type=int, default=None)
    parser.add_argument("--keep_aspect", action="store_true")
    parser.add_argument("--pad_value", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    log_level = get_nested(config, "logging.level", "INFO")
    setup_logging(log_level)

    layout = ensure_layout(build_video_layout(args.output_dir))
    copied_video = copy_raw_video(args.input_video, layout.raw_video_path)

    target_width = args.target_width or get_nested(config, "video.target_width")
    target_height = args.target_height or get_nested(config, "video.target_height")
    keep_aspect = args.keep_aspect or bool(get_nested(config, "video.keep_aspect", False))
    pad_value = (
        args.pad_value
        if args.pad_value is not None
        else int(get_nested(config, "video.pad_value", 0))
    )

    video_info = extract_video_frames(
        copied_video,
        layout.frame_dir,
        target_width=target_width,
        target_height=target_height,
        keep_aspect=keep_aspect,
        pad_value=pad_value,
    )

    metadata = build_metadata(
        video_name=infer_video_name(args.output_dir, args.input_video),
        fps=video_info.fps,
        width=video_info.width,
        height=video_info.height,
        num_frames=video_info.num_frames,
        source_video=copied_video,
        frame_dir=layout.frame_dir,
        sam2_mask_dir=layout.sam2_mask_dir,
        baseline_mask_dir=layout.baseline_mask_dir,
        base_dir=PROJECT_ROOT,
    )
    write_metadata(layout.metadata_path, metadata)

    LOGGER.info("Input video: %s", copied_video)
    LOGGER.info("Output directory: %s", layout.root)
    LOGGER.info(
        "Extracted frames with resolution %sx%s and %s frames",
        video_info.width,
        video_info.height,
        video_info.num_frames,
    )


if __name__ == "__main__":
    main()
