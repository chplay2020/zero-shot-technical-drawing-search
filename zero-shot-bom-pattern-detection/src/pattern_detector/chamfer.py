from __future__ import annotations

from typing import List, Tuple

import cv2
import numpy as np


def compute_gradient_orientation(gray_or_binary: np.ndarray) -> np.ndarray:
    """Compute gradient orientation in degrees within [0, 180)."""
    if gray_or_binary is None or gray_or_binary.size == 0:
        raise ValueError("Empty image provided to compute_gradient_orientation")
    if gray_or_binary.ndim != 2:
        raise ValueError("compute_gradient_orientation expects a 2D image")
    img = gray_or_binary.astype(np.float32)
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    angles = cv2.phase(gx, gy, angleInDegrees=True)
    return np.mod(angles, 180.0)


def quantize_orientation(orientation: np.ndarray, num_bins: int = 8) -> np.ndarray:
    """Quantize orientation angles into bins [0, num_bins-1]."""
    if num_bins <= 0:
        raise ValueError("num_bins must be positive")
    bins = np.floor(orientation / (180.0 / num_bins)).astype(np.int32)
    return np.clip(bins, 0, num_bins - 1)


def extract_oriented_edge_points(
    edge: np.ndarray, orientation_bins: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract edge points and their orientation bins."""
    if edge is None or edge.size == 0:
        raise ValueError("Empty edge map provided to extract_oriented_edge_points")
    if edge.shape[:2] != orientation_bins.shape[:2]:
        raise ValueError("edge and orientation_bins must have the same shape")
    ys, xs = np.where(edge > 0)
    if xs.size == 0:
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0,), dtype=np.int32)
    points = np.column_stack([xs, ys]).astype(np.float32)
    bins = orientation_bins[ys, xs].astype(np.int32)
    return points, bins


def build_directional_distance_transforms(
    edge: np.ndarray, orientation_bins: np.ndarray, num_bins: int = 8
) -> List[np.ndarray]:
    """Build distance transforms per orientation bin."""
    if edge is None or edge.size == 0:
        raise ValueError("Empty edge map provided to build_directional_distance_transforms")
    if num_bins <= 0:
        raise ValueError("num_bins must be positive")
    h, w = edge.shape[:2]
    dts: List[np.ndarray] = []
    edge_mask = edge > 0
    for b in range(num_bins):
        bin_mask = edge_mask & (orientation_bins == b)
        dt_input = np.full((h, w), 255, dtype=np.uint8)
        dt_input[bin_mask] = 0
        dist = cv2.distanceTransform(dt_input, cv2.DIST_L2, 3)
        dts.append(dist.astype(np.float32))
    return dts


def transform_points(
    points: np.ndarray,
    center: Tuple[float, float],
    scale: float,
    rotation_deg: float,
) -> np.ndarray:
    """Apply scale and rotation to points around a center."""
    if points.size == 0:
        return points.astype(np.float32)
    cx, cy = center
    theta = np.deg2rad(rotation_deg)
    cos_t = float(np.cos(theta))
    sin_t = float(np.sin(theta))
    shifted = points - np.array([cx, cy], dtype=np.float32)
    rot = np.array([[cos_t, -sin_t], [sin_t, cos_t]], dtype=np.float32)
    transformed = (shifted @ rot.T) * float(scale)
    return transformed + np.array([cx, cy], dtype=np.float32)


def directional_chamfer_distance(
    points: np.ndarray,
    point_bins: np.ndarray,
    distance_transforms: List[np.ndarray],
    x_offset: int,
    y_offset: int,
    soft_bins: bool = True,
) -> float:
    """Compute directional chamfer distance for points against DTs."""
    if points.size == 0:
        return float("inf")
    if len(distance_transforms) == 0:
        raise ValueError("distance_transforms must not be empty")

    h, w = distance_transforms[0].shape[:2]
    num_bins = len(distance_transforms)
    penalty = float(max(h, w)) * 2.0

    total = 0.0
    for (x, y), b in zip(points, point_bins):
        xi = int(round(x + x_offset))
        yi = int(round(y + y_offset))
        if xi < 0 or yi < 0 or xi >= w or yi >= h:
            total += penalty
            continue
        if soft_bins:
            b0 = int(b) % num_bins
            b1 = (b0 - 1) % num_bins
            b2 = (b0 + 1) % num_bins
            d0 = distance_transforms[b0][yi, xi]
            d1 = distance_transforms[b1][yi, xi]
            d2 = distance_transforms[b2][yi, xi]
            total += float(min(d0, d1, d2))
        else:
            b0 = int(b) % num_bins
            total += float(distance_transforms[b0][yi, xi])
    return total / float(max(points.shape[0], 1))


def chamfer_distance_to_score(distance: float, tau: float = 4.0) -> float:
    """Convert chamfer distance to a confidence score in [0, 1]."""
    score = float(np.exp(-distance / max(tau, 1e-6)))
    return float(np.clip(score, 0.0, 1.0))


def distance_transform(edge_map: np.ndarray) -> np.ndarray:
    """Compute distance transform on inverted edge map (legacy API)."""
    if edge_map is None or edge_map.size == 0:
        raise ValueError("Empty edge map provided to distance_transform")
    dt_input = np.full(edge_map.shape[:2], 255, dtype=np.uint8)
    dt_input[edge_map > 0] = 0
    dist = cv2.distanceTransform(dt_input, cv2.DIST_L2, 3)
    return dist.astype(np.float32)


def orientation_bins(gray: np.ndarray, num_bins: int) -> np.ndarray:
    """Compute orientation bins in [0, num_bins-1] for each pixel (legacy API)."""
    orientation = compute_gradient_orientation(gray)
    return quantize_orientation(orientation, num_bins=num_bins)


def chamfer_score(
    query_edges: np.ndarray,
    query_orient: np.ndarray,
    dist_transform_map: np.ndarray,
    target_orient: np.ndarray,
    box: Tuple[int, int, int, int],
    num_bins: int,
) -> float:
    """Compute a chamfer-style distance for a candidate box (legacy API)."""
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
