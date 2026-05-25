from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np


def rotate_image_keep_bounds(img: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rotate image while keeping all content within bounds."""
    h, w = img.shape[:2]
    center = (w / 2.0, h / 2.0)
    mat = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    cos = abs(mat[0, 0])
    sin = abs(mat[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))
    mat[0, 2] += (new_w / 2.0) - center[0]
    mat[1, 2] += (new_h / 2.0) - center[1]
    return cv2.warpAffine(img, mat, (new_w, new_h), flags=cv2.INTER_LINEAR)


def bbox_iou(box_a: Tuple[int, int, int, int], box_b: Tuple[int, int, int, int]) -> float:
    """Compute IoU between two boxes (x, y, w, h)."""
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b
    ax2 = ax + aw
    ay2 = ay + ah
    bx2 = bx + bw
    by2 = by + bh
    inter_x1 = max(ax, bx)
    inter_y1 = max(ay, by)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    area_a = max(0, aw) * max(0, ah)
    area_b = max(0, bw) * max(0, bh)
    union = area_a + area_b - inter_area
    if union <= 0:
        return 0.0
    return inter_area / float(union)


def aspect_ratio(box: Tuple[int, int, int, int]) -> float:
    """Compute aspect ratio w/h for a box."""
    _, _, w, h = box
    if h <= 0:
        return 0.0
    return w / float(h)
