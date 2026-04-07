from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.config import get_nested, load_config
from src.common.logging_utils import setup_logging
from src.common.paths import build_video_layout, ensure_layout, infer_video_name
from src.detection.detector import DetectorConfig, build_detector
from src.inpainting.baseline_inpaint import InpaintConfig, run_inpainting
from src.io.frame_io import list_frame_paths, read_frame
from src.io.metadata import build_metadata, load_metadata, write_metadata
from src.io.video_io import compose_video_from_frames, copy_raw_video, extract_video_frames
from src.mask_refine.postprocess import PostprocessConfig, postprocess_mask
from src.motion.lk_motion_filter import LKMotionConfig, filter_dynamic_region
from src.validation.sanity_checks import assert_mask_binary, assert_matching_dirs, summarize_image_dir
from src.visualization.overlay import export_overlay_video

LOGGER = logging.getLogger(__name__)


def _resolve_optional_path(value: str | Path | None, *, base_dir: Path) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    return path if path.is_absolute() else (base_dir / path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the hand-crafted baseline pipeline.")
    parser.add_argument("--input_video", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--save_overlay", action="store_true")
    return parser.parse_args()


def _load_or_prepare_frames(
    copied_video: Path,
    layout,
    config: dict,
) -> None:
    if list_frame_paths(layout.frame_dir):
        return

    target_width = get_nested(config, "preprocess.target_width")
    target_height = get_nested(config, "preprocess.target_height")
    resize_mode = str(get_nested(config, "preprocess.resize_mode", "keep_aspect"))
    pad_value = int(get_nested(config, "preprocess.pad_value", 0))
    extract_video_frames(
        copied_video,
        layout.frame_dir,
        target_width=int(target_width) if target_width is not None else None,
        target_height=int(target_height) if target_height is not None else None,
        keep_aspect=resize_mode == "keep_aspect",
        pad_value=pad_value,
    )


def main() -> None:
    args = parse_args()
    if not args.input_video.exists():
        raise FileNotFoundError(f"Input video not found: {args.input_video}")
    config = load_config(args.config)
    setup_logging(get_nested(config, "logging.level", "INFO"))

    layout = ensure_layout(build_video_layout(args.output_dir))
    copied_video = copy_raw_video(args.input_video, layout.raw_video_path)
    _load_or_prepare_frames(copied_video, layout, config)

    detector_config = DetectorConfig(
        backend=str(get_nested(config, "baseline.detector_backend", "maskrcnn")),
        weights=(
            str(
                _resolve_optional_path(
                    get_nested(config, "baseline.detector_weights"),
                    base_dir=PROJECT_ROOT,
                )
            )
            if get_nested(config, "baseline.detector_weights") is not None
            else None
        ),
        target_classes=tuple(get_nested(config, "baseline.target_classes", ["person", "car"])),
        score_threshold=float(get_nested(config, "baseline.score_threshold", 0.5)),
        mask_threshold=float(get_nested(config, "baseline.mask_threshold", 0.5)),
        device=args.device,
    )
    detector = build_detector(detector_config)

    lk_config = LKMotionConfig(
        win_size=int(get_nested(config, "motion.lk_win_size", 21)),
        max_level=int(get_nested(config, "motion.lk_max_level", 3)),
        criteria_count=int(get_nested(config, "motion.lk_criteria_count", 30)),
        criteria_eps=float(get_nested(config, "motion.lk_criteria_eps", 0.01)),
        motion_threshold=float(get_nested(config, "motion.motion_threshold", 1.0)),
        min_points=int(get_nested(config, "motion.min_points", 8)),
    )
    refine_config = PostprocessConfig(
        binary_threshold=float(get_nested(config, "mask_refine.binary_threshold", 0.5)),
        dilation_kernel=int(get_nested(config, "mask_refine.dilation_kernel", 7)),
        dilation_iterations=int(get_nested(config, "mask_refine.dilation_iterations", 1)),
        min_component_area=int(get_nested(config, "mask_refine.min_component_area", 0)),
        keep_largest_component=bool(get_nested(config, "mask_refine.keep_largest_component", False)),
        fill_holes=bool(get_nested(config, "mask_refine.fill_holes", True)),
    )
    inpaint_config = InpaintConfig(
        method=str(get_nested(config, "inpaint.method", "telea")),
        radius=float(get_nested(config, "inpaint.radius", 3)),
    )

    frame_paths = list_frame_paths(layout.frame_dir)
    previous_frame = None
    use_lk_motion_filter = bool(get_nested(config, "baseline.use_lk_motion_filter", True))
    fail_if_no_detections = bool(get_nested(config, "runtime.fail_if_no_detections", False))
    fail_if_empty_masks = bool(get_nested(config, "runtime.fail_if_empty_masks", False))
    total_foreground_pixels = 0
    frames_with_detections = 0
    for frame_path in frame_paths:
        frame = read_frame(frame_path, flags=cv2.IMREAD_COLOR)
        candidate_mask = detector.predict_mask(frame)
        if np.count_nonzero(candidate_mask) > 0:
            frames_with_detections += 1
        motion_mask = (
            filter_dynamic_region(previous_frame, frame, candidate_mask, lk_config)
            if use_lk_motion_filter
            else candidate_mask
        )
        refined_mask = postprocess_mask(motion_mask, refine_config)
        inpainted = run_inpainting(frame, refined_mask, inpaint_config)
        total_foreground_pixels += int(np.count_nonzero(refined_mask))

        mask_output = layout.baseline_mask_dir / frame_path.name
        result_output = layout.baseline_result_dir / frame_path.name
        if not cv2.imwrite(str(mask_output), refined_mask):
            raise IOError(f"Failed to write baseline mask: {mask_output}")
        if not cv2.imwrite(str(result_output), inpainted):
            raise IOError(f"Failed to write inpainted result: {result_output}")
        previous_frame = frame

    if fail_if_empty_masks and total_foreground_pixels == 0:
        raise RuntimeError("Baseline generated only empty masks; runtime.fail_if_empty_masks is true.")
    if fail_if_no_detections and frames_with_detections == 0:
        raise RuntimeError("Baseline detector produced no candidate detections; runtime.fail_if_no_detections is true.")

    metadata = (
        load_metadata(layout.metadata_path)
        if layout.metadata_path.exists()
        else build_metadata(
            video_name=infer_video_name(args.output_dir, args.input_video),
            fps=25.0,
            width=0,
            height=0,
            num_frames=0,
            source_video=copied_video,
            frame_dir=layout.frame_dir,
            sam2_mask_dir=layout.sam2_mask_dir,
            baseline_mask_dir=layout.baseline_mask_dir,
            base_dir=PROJECT_ROOT,
        )
    )
    frame_summary = summarize_image_dir(layout.frame_dir)
    metadata.update(
        {
            "video_name": infer_video_name(args.output_dir, args.input_video),
            "width": frame_summary.frame_size[0],
            "height": frame_summary.frame_size[1],
            "num_frames": frame_summary.frame_count,
            "source_video": str(Path(metadata["source_video"])),
            "frame_dir": str(Path(metadata["frame_dir"])),
            "sam2_mask_dir": str(Path(metadata["sam2_mask_dir"])),
            "baseline_mask_dir": str(layout.baseline_mask_dir.relative_to(PROJECT_ROOT)),
        }
    )
    write_metadata(layout.metadata_path, metadata)

    assert_matching_dirs(layout.frame_dir, layout.baseline_mask_dir)
    assert_matching_dirs(layout.frame_dir, layout.baseline_result_dir)
    assert_mask_binary(layout.baseline_mask_dir)

    result_video_path = layout.video_dir / "baseline_result.mp4"
    fps = float(metadata.get("fps", 25.0) or 25.0)
    compose_video_from_frames(layout.baseline_result_dir, result_video_path, fps=fps)

    save_overlay = args.save_overlay or bool(get_nested(config, "baseline.save_overlay", True))
    if save_overlay:
        overlay_path = layout.overlay_dir / "baseline_overlay.mp4"
        export_overlay_video(layout.frame_dir, layout.baseline_mask_dir, overlay_path, fps=fps)
        LOGGER.info("Saved overlay video to %s", overlay_path)

    LOGGER.info("Device: %s", args.device)
    LOGGER.info("Input video: %s", copied_video)
    LOGGER.info("Baseline masks: %s", layout.baseline_mask_dir)
    LOGGER.info("Baseline results: %s", layout.baseline_result_dir)
    LOGGER.info("Baseline video: %s", result_video_path)


if __name__ == "__main__":
    main()
