from __future__ import annotations

import numpy as np


def edge_iou(query_edges: np.ndarray, target_edges: np.ndarray) -> float:
    """Compute edge IoU between two binary edge maps."""
    q = query_edges > 0
    t = target_edges > 0
    inter = np.logical_and(q, t).sum()
    union = np.logical_or(q, t).sum()
    if union == 0:
        return 0.0
    return float(inter) / float(union)


def density_ratio_score(query_density: float, candidate_density: float) -> float:
    """Score density ratio closeness to 1.0."""
    ratio = candidate_density / max(query_density, 1e-6)
    return float(np.exp(-abs(ratio - 1.0)))


def aspect_ratio_score(query_ar: float, candidate_ar: float) -> float:
    """Score aspect ratio closeness to 1.0."""
    if query_ar <= 0 or candidate_ar <= 0:
        return 0.0
    return float(np.exp(-abs(np.log(candidate_ar / query_ar))))


def fuse_scores(
    chamfer_dist: float,
    edge_iou_score: float,
    density_score: float,
    ar_score: float,
    chamfer_tau: float,
) -> float:
    """Fuse multiple scores into a confidence value in [0, 1]."""
    chamfer_sim = float(np.exp(-chamfer_dist / max(chamfer_tau, 1e-6)))
    score = 0.5 * chamfer_sim + 0.2 * edge_iou_score + 0.2 * density_score + 0.1 * ar_score
    return float(np.clip(score, 0.0, 1.0))
