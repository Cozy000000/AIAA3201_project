from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


@dataclass(frozen=True)
class DetectorConfig:
    backend: str = "maskrcnn"
    weights: str | None = None
    target_classes: tuple[str, ...] = ("person", "car")
    score_threshold: float = 0.5
    mask_threshold: float = 0.5
    device: str = "cpu"


class BaseDetector:
    def predict_mask(self, frame: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class MotionSaliencyDetector(BaseDetector):
    """Fallback detector based on background subtraction when model stacks are unavailable."""

    def __init__(self) -> None:
        self._subtractor = cv2.createBackgroundSubtractorMOG2(
            history=200,
            varThreshold=16,
            detectShadows=False,
        )

    def predict_mask(self, frame: np.ndarray) -> np.ndarray:
        mask = self._subtractor.apply(frame)
        _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        return binary.astype(np.uint8)


class TorchvisionMaskRCNNDetector(BaseDetector):
    COCO_INSTANCE_CATEGORY_NAMES = (
        "__background__",
        "person",
        "bicycle",
        "car",
        "motorcycle",
        "airplane",
        "bus",
        "train",
        "truck",
        "boat",
    )

    def __init__(self, config: DetectorConfig) -> None:
        import torch
        from torchvision.models.detection import MaskRCNN_ResNet50_FPN_V2_Weights, maskrcnn_resnet50_fpn_v2

        self._torch = torch
        self._device = torch.device(config.device)
        weights = MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT
        self._model = maskrcnn_resnet50_fpn_v2(weights=weights)
        self._model.to(self._device)
        self._model.eval()
        self._target_classes = set(config.target_classes)
        self._score_threshold = config.score_threshold
        self._mask_threshold = config.mask_threshold

    def predict_mask(self, frame: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensor = self._torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        tensor = tensor.to(self._device)
        with self._torch.no_grad():
            output = self._model([tensor])[0]

        height, width = frame.shape[:2]
        mask = np.zeros((height, width), dtype=np.uint8)
        scores = output["scores"].detach().cpu().numpy()
        labels = output["labels"].detach().cpu().numpy()
        masks = output["masks"].detach().cpu().numpy()
        for score, label, candidate_mask in zip(scores, labels, masks):
            if float(score) < self._score_threshold:
                continue
            class_name = self.COCO_INSTANCE_CATEGORY_NAMES[int(label)]
            if class_name not in self._target_classes:
                continue
            candidate_binary = (candidate_mask[0] >= self._mask_threshold).astype(np.uint8) * 255
            mask = np.maximum(mask, candidate_binary.astype(np.uint8))
        return mask


class UltralyticsYOLOSegDetector(BaseDetector):
    def __init__(self, config: DetectorConfig) -> None:
        from ultralytics import YOLO

        weights = config.weights or "yolov8n-seg.pt"
        self._model = YOLO(weights)
        self._target_classes = set(config.target_classes)
        self._score_threshold = config.score_threshold

    def predict_mask(self, frame: np.ndarray) -> np.ndarray:
        results = self._model.predict(frame, verbose=False)
        if not results:
            return np.zeros(frame.shape[:2], dtype=np.uint8)

        result = results[0]
        if result.masks is None or result.boxes is None:
            return np.zeros(frame.shape[:2], dtype=np.uint8)

        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        classes = result.boxes.cls.detach().cpu().numpy().astype(int)
        scores = result.boxes.conf.detach().cpu().numpy()
        names: dict[int, str] = result.names

        for index, (class_id, score) in enumerate(zip(classes, scores)):
            if float(score) < self._score_threshold:
                continue
            class_name = names.get(int(class_id), str(class_id))
            if class_name not in self._target_classes:
                continue
            candidate = result.masks.data[index].detach().cpu().numpy()
            candidate = cv2.resize(
                candidate.astype(np.float32),
                (frame.shape[1], frame.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
            candidate = (candidate > 0.5).astype(np.uint8) * 255
            mask = np.maximum(mask, candidate.astype(np.uint8))
        return mask


def _has_module(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def build_detector(config: DetectorConfig) -> BaseDetector:
    """Build the requested detector backend with a stable fallback."""
    backend = config.backend.lower()
    if backend == "yolov8_seg":
        if _has_module("ultralytics"):
            return UltralyticsYOLOSegDetector(config)
        raise RuntimeError(
            "Requested backend 'yolov8_seg' but ultralytics is not installed in the active environment."
        )

    if backend in {"maskrcnn", "mask_rcnn"}:
        if _has_module("torchvision"):
            return TorchvisionMaskRCNNDetector(config)
        raise RuntimeError(
            "Requested backend 'maskrcnn' but torchvision is not installed in the active environment."
        )

    if backend == "motion_saliency":
        return MotionSaliencyDetector()

    raise ValueError(f"Unsupported detector backend: {config.backend}")
