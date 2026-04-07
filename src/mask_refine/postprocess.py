from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class PostprocessConfig:
    binary_threshold: float = 0.5
    dilation_kernel: int = 5
    dilation_iterations: int = 1
    min_component_area: int = 0
    keep_largest_component: bool = False
    fill_holes: bool = True


def _ensure_uint8_mask(mask: np.ndarray, threshold: float) -> np.ndarray:
    if mask.ndim == 3:
        mask = np.squeeze(mask)
    if mask.ndim != 2:
        raise ValueError(f"Expected a 2D mask array, got shape {mask.shape}")

    if mask.dtype == np.uint8 and set(np.unique(mask)).issubset({0, 255}):
        return mask.copy()

    mask_float = mask.astype(np.float32)
    binary = (mask_float > threshold).astype(np.uint8) * 255
    return binary


def _remove_small_components(mask: np.ndarray, min_component_area: int) -> np.ndarray:
    if min_component_area <= 0:
        return mask

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    filtered = np.zeros_like(mask)
    for label_index in range(1, num_labels):
        area = stats[label_index, cv2.CC_STAT_AREA]
        if area >= min_component_area:
            filtered[labels == label_index] = 255
    return filtered


def _keep_largest_component(mask: np.ndarray) -> np.ndarray:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return mask

    largest_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    largest = np.zeros_like(mask)
    largest[labels == largest_label] = 255
    return largest


def _fill_holes(mask: np.ndarray) -> np.ndarray:
    contours, _ = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    filled = mask.copy()
    for contour in contours:
        cv2.drawContours(filled, [contour], -1, color=255, thickness=cv2.FILLED)
    return filled


def postprocess_mask(mask: np.ndarray, config: PostprocessConfig) -> np.ndarray:
    """Convert one raw mask/logit map into a binary 0/255 mask."""
    refined = _ensure_uint8_mask(mask, threshold=config.binary_threshold)

    if config.dilation_kernel > 1 and config.dilation_iterations > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (config.dilation_kernel, config.dilation_kernel),
        )
        refined = cv2.dilate(refined, kernel, iterations=config.dilation_iterations)

    refined = _remove_small_components(refined, config.min_component_area)

    if config.keep_largest_component:
        refined = _keep_largest_component(refined)

    if config.fill_holes:
        refined = _fill_holes(refined)

    return refined.astype(np.uint8)
