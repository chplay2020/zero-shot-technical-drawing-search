from __future__ import annotations

from typing import Dict

import numpy as np


def _clamp01(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def edge_iou_score(pattern_edge: np.ndarray, candidate_edge: np.ndarray) -> float:
    """Compute edge IoU between two binary edge maps."""
    if pattern_edge is None or candidate_edge is None:
        return 0.0
    p = pattern_edge > 0
    c = candidate_edge > 0
    inter = np.logical_and(p, c).sum()
    union = np.logical_or(p, c).sum()
    if union == 0:
        return 0.0
    return _clamp01(float(inter) / float(union))


def density_consistency_score(candidate_edge_count: float, pattern_edge_count: float) -> float:
    """Score density consistency between candidate and pattern in [0, 1]."""
    if pattern_edge_count <= 0:
        return 0.0
    ratio = candidate_edge_count / max(pattern_edge_count, 1e-6)
    return _clamp01(float(np.exp(-abs(ratio - 1.0))))


def aspect_ratio_score(candidate_w: int, candidate_h: int, pattern_w: int, pattern_h: int) -> float:
    """Score aspect ratio similarity in [0, 1]."""
    if candidate_w <= 0 or candidate_h <= 0 or pattern_w <= 0 or pattern_h <= 0:
        return 0.0
    cand_ar = float(candidate_w) / float(candidate_h)
    pat_ar = float(pattern_w) / float(pattern_h)
    return _clamp01(float(np.exp(-abs(np.log(cand_ar / pat_ar)))))


def fuse_scores(
    chamfer_score: float,
    edge_iou: float,
    density: float,
    aspect: float,
    weights: Dict[str, float] | None = None,
) -> float:
    """Fuse sub-scores into a confidence value in [0, 1]."""
    if weights is None:
        weights = {"chamfer": 0.5, "edge_iou": 0.2, "density": 0.2, "aspect": 0.1}
    total = (
        float(weights.get("chamfer", 0.0)) * chamfer_score
        + float(weights.get("edge_iou", 0.0)) * edge_iou
        + float(weights.get("density", 0.0)) * density
        + float(weights.get("aspect", 0.0)) * aspect
    )
    return _clamp01(total)


def edge_iou(query_edges: np.ndarray, target_edges: np.ndarray) -> float:
    """Backward-compatible wrapper for edge_iou_score."""
    return edge_iou_score(query_edges, target_edges)


def density_ratio_score(query_density: float, candidate_density: float) -> float:
    """Backward-compatible wrapper using density ratio."""
    if query_density <= 0:
        return 0.0
    ratio = candidate_density / max(query_density, 1e-6)
    return _clamp01(float(np.exp(-abs(ratio - 1.0))))
