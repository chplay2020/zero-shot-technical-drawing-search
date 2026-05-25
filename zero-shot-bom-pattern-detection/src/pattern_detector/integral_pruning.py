from __future__ import annotations

from typing import Iterable, List, Tuple

import numpy as np


Candidate = Tuple[int, int, int, int, float, float]


def integral_image(edge_map: np.ndarray) -> np.ndarray:
    """Compute integral image from binary edge map."""
    return edge_map.cumsum(axis=0).cumsum(axis=1).astype(np.float32)


def _sum_in_box(integral: np.ndarray, x: int, y: int, w: int, h: int) -> float:
    x2 = x + w - 1
    y2 = y + h - 1
    total = integral[y2, x2]
    left = integral[y2, x - 1] if x > 0 else 0.0
    up = integral[y - 1, x2] if y > 0 else 0.0
    corner = integral[y - 1, x - 1] if (x > 0 and y > 0) else 0.0
    return float(total - left - up + corner)


def prune_by_density(
    candidates: Iterable[Candidate],
    integral: np.ndarray,
    query_edge_density: float,
    min_ratio: float,
    max_ratio: float,
) -> List[Candidate]:
    """Filter candidates by edge density ratio to query."""
    kept: List[Candidate] = []
    for x, y, w, h, scale, rot in candidates:
        if w <= 0 or h <= 0:
            continue
        edge_count = _sum_in_box(integral, x, y, w, h)
        density = edge_count / float(w * h)
        ratio = density / max(query_edge_density, 1e-6)
        if min_ratio <= ratio <= max_ratio:
            kept.append((x, y, w, h, scale, rot))
    return kept
