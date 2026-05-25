from __future__ import annotations

from typing import Iterable, Tuple

import cv2
import numpy as np


def draw_detections(
    image_bgr: np.ndarray,
    boxes: Iterable[Tuple[int, int, int, int]],
    scores: Iterable[float],
) -> np.ndarray:
    """Draw detection boxes and scores on the image."""
    vis = image_bgr.copy()
    for (x, y, w, h), score in zip(boxes, scores):
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 200, 0), 2)
        cv2.putText(
            vis,
            f"{score:.2f}",
            (x, max(0, y - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 200, 0),
            1,
            cv2.LINE_AA,
        )
    return vis
