from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class LKMotionConfig:
    win_size: int = 21
    max_level: int = 3
    criteria_count: int = 30
    criteria_eps: float = 0.01
    motion_threshold: float = 1.0
    min_points: int = 8


def filter_dynamic_region(
    previous_frame: np.ndarray | None,
    current_frame: np.ndarray,
    candidate_mask: np.ndarray,
    config: LKMotionConfig,
) -> np.ndarray:
    """Keep only regions whose sparse optical flow indicates motion."""
    if previous_frame is None:
        return candidate_mask

    if candidate_mask.ndim != 2:
        raise ValueError(f"Expected a 2D candidate mask, got {candidate_mask.shape}")

    prev_gray = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

    feature_mask = candidate_mask.copy()
    points = cv2.goodFeaturesToTrack(
        prev_gray,
        maxCorners=300,
        qualityLevel=0.01,
        minDistance=5,
        mask=feature_mask,
    )
    if points is None or len(points) < config.min_points:
        return candidate_mask

    next_points, status, _ = cv2.calcOpticalFlowPyrLK(
        prev_gray,
        curr_gray,
        points,
        None,
        winSize=(config.win_size, config.win_size),
        maxLevel=config.max_level,
        criteria=(
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            config.criteria_count,
            config.criteria_eps,
        ),
    )
    if next_points is None or status is None:
        return np.zeros_like(candidate_mask)

    valid_previous = points[status.flatten() == 1]
    valid_next = next_points[status.flatten() == 1]
    if len(valid_previous) < config.min_points:
        return np.zeros_like(candidate_mask)

    magnitudes = np.linalg.norm(valid_next - valid_previous, axis=1)
    dynamic_mask = np.zeros_like(candidate_mask)
    for point, magnitude in zip(valid_previous, magnitudes):
        if float(magnitude) >= config.motion_threshold:
            x, y = point.ravel()
            cv2.circle(dynamic_mask, (int(round(x)), int(round(y))), 8, 255, thickness=-1)

    filtered = cv2.bitwise_and(candidate_mask, dynamic_mask)
    return filtered.astype(np.uint8)
