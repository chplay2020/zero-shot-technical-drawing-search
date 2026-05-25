from __future__ import annotations

from typing import List, Tuple

import numpy as np


Box = Tuple[int, int, int, int]


def nms(boxes: List[Box], scores: List[float], iou_threshold: float) -> List[int]:
    """Perform non-maximum suppression and return kept indices."""
    if not boxes:
        return []
    x1 = np.array([b[0] for b in boxes], dtype=np.float32)
    y1 = np.array([b[1] for b in boxes], dtype=np.float32)
    x2 = np.array([b[0] + b[2] for b in boxes], dtype=np.float32)
    y2 = np.array([b[1] + b[3] for b in boxes], dtype=np.float32)
    scores_arr = np.array(scores, dtype=np.float32)

    order = scores_arr.argsort()[::-1]
    keep: List[int] = []

    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        area_i = (x2[i] - x1[i]) * (y2[i] - y1[i])
        area_o = (x2[order[1:]] - x1[order[1:]]) * (y2[order[1:]] - y1[order[1:]])
        iou = inter / (area_i + area_o - inter + 1e-6)

        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]

    return keep
