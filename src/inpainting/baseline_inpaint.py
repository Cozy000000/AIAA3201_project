from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class InpaintConfig:
    method: str = "telea"
    radius: float = 3.0


def run_inpainting(frame: np.ndarray, mask: np.ndarray, config: InpaintConfig) -> np.ndarray:
    """Run the classical OpenCV inpainting baseline."""
    method = config.method.lower()
    if method == "telea":
        inpaint_flag = cv2.INPAINT_TELEA
    elif method == "ns":
        inpaint_flag = cv2.INPAINT_NS
    else:
        raise ValueError(f"Unsupported inpaint method: {config.method}")

    return cv2.inpaint(frame, mask, float(config.radius), inpaint_flag)
