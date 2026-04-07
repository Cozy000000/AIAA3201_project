from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.config import get_nested, load_config
from src.common.logging_utils import setup_logging
from src.common.paths import build_video_layout, ensure_layout, infer_video_name
from src.io.frame_io import list_frame_paths
from src.io.metadata import build_metadata, write_metadata
from src.io.video_io import copy_raw_video, extract_video_frames
from src.mask_refine.postprocess import PostprocessConfig
from src.segmentation.sam2_pipeline import (
    OfficialSAM2VideoPipeline,
    SAM2DependencyError,
    load_prompt_bundle,
)
from src.validation.sanity_checks import assert_mask_binary, assert_matching_dirs, summarize_image_dir
from src.visualization.overlay import export_overlay_video

LOGGER = logging.getLogger(__name__)


def _resolve_optional_path(value: str | Path | None, *, base_dir: Path) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    return path if path.is_absolute() else (base_dir / path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the SAM2 mask extraction pipeline.")
    parser.add_argument("--input_video", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--save_overlay", action="store_true")
    parser.add_argument("--prompt_json", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input_video.exists():
        raise FileNotFoundError(f"Input video not found: {args.input_video}")
    config = load_config(args.config)
    setup_logging(get_nested(config, "logging.level", "INFO"))

    layout = ensure_layout(build_video_layout(args.output_dir))
    copied_video = copy_raw_video(args.input_video, layout.raw_video_path)

    frame_paths = list_frame_paths(layout.frame_dir)
    if not frame_paths:
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
        frame_paths = list_frame_paths(layout.frame_dir)

    if not frame_paths:
        raise FileNotFoundError(f"Failed to prepare frames in {layout.frame_dir}")

    prompt_json = _resolve_optional_path(
        args.prompt_json or get_nested(config, "prompt.prompt_json"),
        base_dir=PROJECT_ROOT,
    )
    require_prompt_json = bool(get_nested(config, "prompt.require_prompt_json", True))
    automatic_mode = bool(get_nested(config, "sam2.automatic_mode", False))
    if prompt_json is None and require_prompt_json and not automatic_mode:
        raise ValueError(
            "SAM2 prompt JSON is required for this configuration. "
            "Pass --prompt_json or set prompt.prompt_json in the YAML."
        )
    if automatic_mode:
        raise NotImplementedError(
            "automatic_mode is reserved but not implemented in this project slice yet."
        )

    prompts = load_prompt_bundle(
        prompt_json,
        video_name=infer_video_name(args.output_dir, args.input_video),
    )

    postprocess_config = PostprocessConfig(
        binary_threshold=float(get_nested(config, "postprocess.binary_threshold", 0.5)),
        dilation_kernel=int(get_nested(config, "postprocess.dilation_kernel", 5)),
        dilation_iterations=int(get_nested(config, "postprocess.dilation_iterations", 1)),
        min_component_area=int(get_nested(config, "postprocess.min_component_area", 0)),
        keep_largest_component=bool(get_nested(config, "postprocess.keep_largest_component", False)),
        fill_holes=bool(get_nested(config, "postprocess.fill_holes", True)),
    )

    model_id = get_nested(config, "sam2.model_id")
    model_config_path = _resolve_optional_path(
        get_nested(config, "sam2.model_config_path"),
        base_dir=PROJECT_ROOT,
    )
    checkpoint_path = _resolve_optional_path(
        get_nested(config, "sam2.checkpoint_path"),
        base_dir=PROJECT_ROOT,
    )
    if not model_id and (not model_config_path or not checkpoint_path):
        raise ValueError(
            "Set either sam2.model_id, or both sam2.model_config_path and sam2.checkpoint_path."
        )
    if model_config_path is not None and not model_config_path.exists():
        raise FileNotFoundError(f"SAM2 config file not found: {model_config_path}")
    if checkpoint_path is not None and not checkpoint_path.exists():
        raise FileNotFoundError(f"SAM2 checkpoint not found: {checkpoint_path}")

    pipeline = OfficialSAM2VideoPipeline(
        model_id=model_id,
        model_config_path=model_config_path,
        checkpoint_path=checkpoint_path,
        device=args.device or str(get_nested(config, "sam2.device", "cuda")),
        postprocess=postprocess_config,
    )

    try:
        masks = pipeline.segment_video(frame_dir=layout.frame_dir, prompts=prompts)
    except SAM2DependencyError as error:
        raise RuntimeError(
            f"SAM2 runtime initialization failed: {error}"
        ) from error

    for frame_index, frame_path in enumerate(frame_paths):
        output_mask_path = layout.sam2_mask_dir / frame_path.name
        success = cv2.imwrite(str(output_mask_path), masks[frame_index])
        if not success:
            raise IOError(f"Failed to write mask: {output_mask_path}")

    frame_summary = summarize_image_dir(layout.frame_dir)
    metadata = build_metadata(
        video_name=infer_video_name(args.output_dir, args.input_video),
        fps=float(get_nested(config, "preprocess.target_fps", 0.0) or 0.0),
        width=frame_summary.frame_size[0],
        height=frame_summary.frame_size[1],
        num_frames=frame_summary.frame_count,
        source_video=copied_video,
        frame_dir=layout.frame_dir,
        sam2_mask_dir=layout.sam2_mask_dir,
        baseline_mask_dir=layout.baseline_mask_dir,
        base_dir=PROJECT_ROOT,
    )
    if metadata["fps"] == 0.0 and layout.metadata_path.exists():
        from src.io.metadata import load_metadata

        existing_metadata = load_metadata(layout.metadata_path)
        metadata["fps"] = float(existing_metadata.get("fps", 0.0))
    write_metadata(layout.metadata_path, metadata)

    assert_matching_dirs(layout.frame_dir, layout.sam2_mask_dir)
    assert_mask_binary(layout.sam2_mask_dir)

    save_overlay = args.save_overlay or bool(get_nested(config, "sam2.save_overlay", True))
    if save_overlay:
        overlay_path = layout.overlay_dir / "sam2_overlay.mp4"
        fps = float(metadata["fps"] or 25.0)
        export_overlay_video(
            layout.frame_dir,
            layout.sam2_mask_dir,
            overlay_path,
            fps=fps,
            codec=str(get_nested(config, "video.codec", "mp4v")),
        )
        LOGGER.info("Saved overlay video to %s", overlay_path)

    LOGGER.info("Device: %s", args.device)
    LOGGER.info("Input video: %s", copied_video)
    LOGGER.info("Output masks: %s", layout.sam2_mask_dir)
    LOGGER.info(
        "Exported %s SAM2 masks at %sx%s",
        frame_summary.frame_count,
        frame_summary.frame_size[0],
        frame_summary.frame_size[1],
    )


if __name__ == "__main__":
    main()
