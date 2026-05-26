from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import cv2
import numpy as np


def draw_detections(image: np.ndarray, detections: List[Dict], thickness: int = 2) -> np.ndarray:
    """Draw detection boxes and confidence scores on the image."""
    if image is None or image.size == 0:
        raise ValueError("Empty image provided to draw_detections")
    vis = image.copy()
    for det in detections:
        bbox = det.get("bbox", None)
        conf = float(det.get("confidence", 0.0))
        if bbox is None or len(bbox) != 4:
            continue
        x, y, w, h = [int(v) for v in bbox]
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 200, 0), thickness)
        cv2.putText(
            vis,
            f"{conf:.2f}",
            (x, max(0, y - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 200, 0),
            1,
            cv2.LINE_AA,
        )
    return vis


def draw_detections_legacy(
    image_bgr: np.ndarray,
    boxes: Iterable[Tuple[int, int, int, int]],
    scores: Iterable[float],
) -> np.ndarray:
    """Backward-compatible wrapper for drawing boxes/scores."""
    dets = [
        {"bbox": list(b), "confidence": float(s), "scale": 1.0, "rotation": 0.0, "scores": {}}
        for b, s in zip(boxes, scores)
    ]
    return draw_detections(image_bgr, dets)
