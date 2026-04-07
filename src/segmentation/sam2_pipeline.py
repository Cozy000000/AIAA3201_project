from __future__ import annotations

import importlib
import json
import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.io.frame_io import list_frame_paths
from src.mask_refine.postprocess import PostprocessConfig, postprocess_mask

LOGGER = logging.getLogger(__name__)


class SAM2DependencyError(RuntimeError):
    """Raised when the local SAM2 runtime is missing or incompatible."""


@dataclass(frozen=True)
class PromptPoint:
    x: float
    y: float
    label: int


@dataclass(frozen=True)
class FramePrompt:
    frame_index: int
    bbox: tuple[float, float, float, float] | None
    points: tuple[PromptPoint, ...]
    object_id: int


def load_prompt_bundle(
    prompt_json_path: str | Path,
    *,
    video_name: str | None = None,
) -> list[FramePrompt]:
    """Load and validate the prompt JSON contract for SAM2 initialization."""
    path = Path(prompt_json_path)
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    prompts = payload.get("prompts")
    if not isinstance(prompts, list) or not prompts:
        raise ValueError(f"Prompt JSON must contain a non-empty 'prompts' list: {path}")

    if video_name is not None and "video_name" in payload and payload["video_name"] != video_name:
        raise ValueError(
            f"Prompt bundle video_name={payload['video_name']} does not match {video_name}"
        )

    parsed: list[FramePrompt] = []
    for index, item in enumerate(prompts, start=1):
        if "frame_index" not in item:
            raise ValueError(f"Prompt #{index} is missing frame_index")
        frame_index = int(item["frame_index"])

        bbox_raw = item.get("bbox")
        bbox: tuple[float, float, float, float] | None = None
        if bbox_raw is not None:
            if not isinstance(bbox_raw, list) or len(bbox_raw) != 4:
                raise ValueError(f"Prompt #{index} bbox must be a 4-element list")
            bbox = tuple(float(value) for value in bbox_raw)

        points_raw = item.get("points", [])
        if not isinstance(points_raw, list):
            raise ValueError(f"Prompt #{index} points must be a list")
        points = tuple(
            PromptPoint(
                x=float(point["x"]),
                y=float(point["y"]),
                label=int(point["label"]),
            )
            for point in points_raw
        )

        if bbox is None and not points:
            raise ValueError(f"Prompt #{index} must define at least bbox or points")

        object_id = int(item.get("object_id", index))
        parsed.append(FramePrompt(frame_index=frame_index, bbox=bbox, points=points, object_id=object_id))

    return parsed


class OfficialSAM2VideoPipeline:
    """Thin adapter around the official SAM2 video predictor API."""

    def __init__(
        self,
        *,
        model_id: str | None = None,
        model_config_path: str | Path | None = None,
        checkpoint_path: str | Path | None = None,
        device: str = "cuda",
        postprocess: PostprocessConfig | None = None,
    ) -> None:
        self.model_id = model_id
        self.model_config_path = str(model_config_path) if model_config_path is not None else None
        self.checkpoint_path = str(checkpoint_path) if checkpoint_path is not None else None
        self.device = device
        self.postprocess = postprocess or PostprocessConfig()
        self._predictor = self._build_predictor()

    def _build_predictor(self) -> Any:
        try:
            build_module = importlib.import_module("sam2.build_sam")
        except ModuleNotFoundError as error:
            raise SAM2DependencyError(
                "SAM2 is not installed in the active environment. "
                "Activate the intended environment and install the official SAM2 package."
            ) from error

        if self.model_id:
            build_hf_fn = getattr(build_module, "build_sam2_video_predictor_hf", None)
            if build_hf_fn is None:
                raise SAM2DependencyError(
                    "The installed sam2 package does not expose build_sam2_video_predictor_hf."
                )
            try:
                return build_hf_fn(self.model_id, device=self.device)
            except ModuleNotFoundError as error:
                raise SAM2DependencyError(
                    "SAM2 HF download mode requires huggingface_hub in the active environment."
                ) from error

        build_fn = getattr(build_module, "build_sam2_video_predictor", None)
        if build_fn is None:
            raise SAM2DependencyError(
                "The installed sam2 package does not expose build_sam2_video_predictor. "
                "Please confirm the official video-predictor build is installed."
            )
        if self.model_config_path is None or self.checkpoint_path is None:
            raise SAM2DependencyError(
                "Local SAM2 mode requires both model_config_path and checkpoint_path."
            )

        try:
            predictor = build_fn(self.model_config_path, self.checkpoint_path, device=self.device)
        except TypeError:
            predictor = build_fn(self.model_config_path, self.checkpoint_path)
        return predictor

    def _add_prompt(self, inference_state: Any, prompt: FramePrompt) -> None:
        points_array = None
        labels_array = None
        if prompt.points:
            points_array = np.array([[point.x, point.y] for point in prompt.points], dtype=np.float32)
            labels_array = np.array([point.label for point in prompt.points], dtype=np.int32)

        if not hasattr(self._predictor, "add_new_points_or_box"):
            raise SAM2DependencyError(
                "The active SAM2 predictor does not support add_new_points_or_box."
            )

        self._predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=prompt.frame_index,
            obj_id=prompt.object_id,
            points=points_array,
            labels=labels_array,
            box=np.array(prompt.bbox, dtype=np.float32) if prompt.bbox is not None else None,
        )

    def _prepare_sam2_frame_dir(self, frame_dir: str | Path) -> Path:
        """Convert the project's PNG frame contract into the JPG sequence SAM2 expects."""
        source_paths = list_frame_paths(frame_dir)
        if not source_paths:
            raise FileNotFoundError(f"No frames found in {frame_dir}")

        temp_dir = Path(tempfile.mkdtemp(prefix="sam2_jpg_", dir=str(Path(frame_dir).parent)))
        for frame_path in source_paths:
            frame = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
            if frame is None:
                raise FileNotFoundError(f"Unable to read frame: {frame_path}")
            jpg_path = temp_dir / f"{frame_path.stem}.jpg"
            success = cv2.imwrite(str(jpg_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            if not success:
                raise IOError(f"Failed to write temporary SAM2 JPG frame: {jpg_path}")
        return temp_dir

    def segment_video(
        self,
        *,
        frame_dir: str | Path,
        prompts: list[FramePrompt],
    ) -> dict[int, np.ndarray]:
        frame_paths = list_frame_paths(frame_dir)
        if not frame_paths:
            raise FileNotFoundError(f"No frames found in {frame_dir}")

        if not hasattr(self._predictor, "init_state"):
            raise SAM2DependencyError("The active SAM2 predictor does not support init_state.")

        sam2_frame_dir = self._prepare_sam2_frame_dir(frame_dir)
        inference_state = self._predictor.init_state(video_path=str(sam2_frame_dir))
        if hasattr(self._predictor, "reset_state"):
            self._predictor.reset_state(inference_state)

        for prompt in prompts:
            if prompt.frame_index < 0 or prompt.frame_index >= len(frame_paths):
                raise ValueError(
                    f"Prompt frame_index {prompt.frame_index} is out of range for {len(frame_paths)} frames"
                )
            self._add_prompt(inference_state, prompt)

        if not hasattr(self._predictor, "propagate_in_video"):
            raise SAM2DependencyError(
                "The active SAM2 predictor does not support propagate_in_video."
            )

        raw_masks: dict[int, np.ndarray] = {}
        for frame_index, object_ids, mask_logits in self._predictor.propagate_in_video(inference_state):
            combined = None
            for object_offset, _ in enumerate(object_ids):
                candidate = mask_logits[object_offset]
                if hasattr(candidate, "detach"):
                    candidate = candidate.detach()
                if hasattr(candidate, "cpu"):
                    candidate = candidate.cpu()
                candidate_array = np.array(candidate)
                candidate_array = np.squeeze(candidate_array)
                if combined is None:
                    combined = candidate_array
                else:
                    combined = np.maximum(combined, candidate_array)

            if combined is None:
                continue
            raw_masks[int(frame_index)] = postprocess_mask(combined, self.postprocess)

        if len(raw_masks) != len(frame_paths):
            LOGGER.warning(
                "SAM2 returned %s masks for %s frames; missing frames will be filled with zeros.",
                len(raw_masks),
                len(frame_paths),
            )

        height, width = cv2.imread(str(frame_paths[0]), cv2.IMREAD_COLOR).shape[:2]
        output_masks: dict[int, np.ndarray] = {}
        for frame_index in range(len(frame_paths)):
            output_masks[frame_index] = raw_masks.get(
                frame_index,
                np.zeros((height, width), dtype=np.uint8),
            )
        return output_masks
