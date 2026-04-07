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
from src.common.paths import build_video_layout
from src.validation.sanity_checks import assert_mask_binary, assert_matching_dirs, summarize_image_dir

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check frame and mask alignment in the standardized layout.")
    parser.add_argument("--input_video", type=Path, default=None)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--save_overlay", action="store_true")
    parser.add_argument("--frames_dir", type=Path, default=None)
    parser.add_argument("--mask_dir", type=Path, action="append", default=None)
    parser.add_argument("--expect_binary_masks", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    setup_logging(get_nested(config, "logging.level", "INFO"))

    layout = build_video_layout(args.output_dir)
    frames_dir = args.frames_dir or layout.frame_dir
    mask_dirs = args.mask_dir or []

    frame_summary = summarize_image_dir(frames_dir)
    LOGGER.info(
        "Reference frames: %s files at %sx%s",
        frame_summary.frame_count,
        frame_summary.frame_size[0],
        frame_summary.frame_size[1],
    )

    for mask_dir in mask_dirs:
        assert_matching_dirs(frames_dir, mask_dir)
        if args.expect_binary_masks:
            assert_mask_binary(mask_dir)
        LOGGER.info("Alignment check passed for %s", mask_dir)


if __name__ == "__main__":
    main()
