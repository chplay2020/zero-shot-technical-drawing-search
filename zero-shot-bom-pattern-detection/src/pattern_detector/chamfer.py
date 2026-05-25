from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np


def distance_transform(edge_map: np.ndarray) -> np.ndarray:
    """Compute distance transform on inverted edge map."""
    inv = 1 - edge_map.astype(np.uint8)
    dist = cv2.distanceTransform(inv, cv2.DIST_L2, 3)
    return dist.astype(np.float32)


def orientation_bins(gray: np.ndarray, num_bins: int) -> np.ndarray:
    """Compute orientation bins in [0, num_bins-1] for each pixel."""
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    angles = cv2.phase(gx, gy, angleInDegrees=True)
    angles = np.mod(angles, 180.0)
    bins = np.floor(angles / (180.0 / num_bins)).astype(np.int32)
    return bins


def chamfer_score(
    query_edges: np.ndarray,
    query_orient: np.ndarray,
    dist_transform_map: np.ndarray,
    target_orient: np.ndarray,
    box: Tuple[int, int, int, int],
    num_bins: int,
) -> float:
    """Compute a chamfer-style distance for a candidate box."""
    x, y, w, h = box
    if w <= 0 or h <= 0:
        return float("inf")

    dist_roi = dist_transform_map[y : y + h, x : x + w]
    orient_roi = target_orient[y : y + h, x : x + w]

    q_edges = cv2.resize(query_edges, (w, h), interpolation=cv2.INTER_NEAREST)
    q_orient = cv2.resize(query_orient, (w, h), interpolation=cv2.INTER_NEAREST)

    edge_idx = np.where(q_edges > 0)
    if edge_idx[0].size == 0:
        return float("inf")

    dist_vals = dist_roi[edge_idx]
    orient_diff = np.abs(q_orient[edge_idx] - orient_roi[edge_idx])
    orient_diff = np.minimum(orient_diff, num_bins - orient_diff)
    orient_penalty = orient_diff.astype(np.float32) / float(max(1, num_bins))

    return float(np.mean(dist_vals + 0.5 * orient_penalty))
