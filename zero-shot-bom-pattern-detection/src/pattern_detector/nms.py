from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np


Box = Tuple[int, int, int, int]


def box_iou(box_a: Box, box_b: Box) -> float:
    """Compute IoU between two boxes in (x, y, w, h)."""
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
    return float(inter_area) / float(union)


def nms_indices(boxes: List[Box], scores: List[float], iou_threshold: float) -> List[int]:
    """Perform NMS on boxes/scores and return kept indices."""
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


def _to_box(bbox: List[int] | Tuple[int, int, int, int]) -> Box | None:
    if len(bbox) != 4:
        return None
    return (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))


def nms(detections: List[Dict], iou_threshold: float = 0.3) -> List[Dict]:
    """Perform NMS for detection dicts using bbox and confidence."""
    if not detections:
        return []
    dets_sorted = sorted(detections, key=lambda d: float(d.get("confidence", 0.0)), reverse=True)
    keep: List[Dict] = []
    for det in dets_sorted:
        bbox = det.get("bbox", None)
        if bbox is None:
            continue
        box = _to_box(bbox)
        if box is None:
            continue
        ok = True
        for k in keep:
            other = _to_box(k["bbox"])
            if other is None:
                continue
            if box_iou(box, other) > iou_threshold:
                ok = False
                break
        if ok:
            keep.append(det)
    return keep
