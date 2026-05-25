from __future__ import annotations

from typing import Iterable, List, Tuple

import numpy as np


Candidate = Tuple[int, int, int, int, float, float]


def build_integral_image(edge: np.ndarray) -> np.ndarray:
    """Build integral image from a binary edge map."""
    if edge is None or edge.size == 0:
        raise ValueError("Empty edge map provided to build_integral_image")
    binary = (edge > 0).astype(np.float32)
    return binary.cumsum(axis=0).cumsum(axis=1)


def window_sum(integral: np.ndarray, x: int, y: int, w: int, h: int) -> float:
    """Compute sum of edges in a window using the integral image."""
    if w <= 0 or h <= 0:
        return 0.0
    x2 = x + w - 1
    y2 = y + h - 1
    total = integral[y2, x2]
    left = integral[y2, x - 1] if x > 0 else 0.0
    up = integral[y - 1, x2] if y > 0 else 0.0
    corner = integral[y - 1, x - 1] if (x > 0 and y > 0) else 0.0
    return float(total - left - up + corner)


def density_score(window_edge_count: float, pattern_edge_count: float) -> float:
    """Score density similarity between window and pattern in [0, 1]."""
    if pattern_edge_count <= 0:
        raise ValueError("pattern_edge_count must be positive")
    ratio = window_edge_count / max(pattern_edge_count, 1e-6)
    return float(np.exp(-abs(ratio - 1.0)))


def is_density_valid(
    window_edge_count: float,
    pattern_edge_count: float,
    min_ratio: float,
    max_ratio: float,
) -> bool:
    """Check if window edge density is within ratio bounds."""
    if pattern_edge_count <= 0:
        return False
    ratio = window_edge_count / max(pattern_edge_count, 1e-6)
    return min_ratio <= ratio <= max_ratio


def integral_image(edge_map: np.ndarray) -> np.ndarray:
    """Backward-compatible wrapper for build_integral_image."""
    return build_integral_image(edge_map)


def _sum_in_box(integral: np.ndarray, x: int, y: int, w: int, h: int) -> float:
    """Backward-compatible wrapper for window_sum."""
    return window_sum(integral, x, y, w, h)


def prune_by_density(
    candidates: Iterable[Candidate],
    integral: np.ndarray,
    query_edge_density: float,
    min_ratio: float,
    max_ratio: float,
) -> List[Candidate]:
    """Filter candidates by edge density ratio to query (legacy API)."""
    kept: List[Candidate] = []
    if query_edge_density <= 0:
        return kept
    for x, y, w, h, scale, rot in candidates:
        if w <= 0 or h <= 0:
            continue
        edge_count = window_sum(integral, x, y, w, h)
        density = edge_count / float(w * h)
        ratio = density / max(query_edge_density, 1e-6)
        if min_ratio <= ratio <= max_ratio:
            kept.append((x, y, w, h, scale, rot))
    return kept
