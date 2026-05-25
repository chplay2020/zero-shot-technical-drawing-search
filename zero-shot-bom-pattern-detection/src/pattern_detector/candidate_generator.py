from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import cv2
import numpy as np

from pattern_detector.integral_pruning import (
    build_integral_image,
    density_score,
    is_density_valid,
    window_sum,
)


@dataclass(frozen=True)
class CandidateBox:
    """Candidate window with density-based scoring."""

    x: int
    y: int
    w: int
    h: int
    scale: float
    rotation: float
    density_score: float
    raw_score: float


def generate_scale_rotation_variants(
    pattern_edge: np.ndarray,
    scales: List[float],
    rotations: List[float],
) -> List[Dict[str, float]]:
    """Generate scale/rotation variants from a pattern edge map."""
    if pattern_edge is None or pattern_edge.size == 0:
        raise ValueError("Empty pattern edge provided to generate_scale_rotation_variants")
    h, w = pattern_edge.shape[:2]
    edge_count = float((pattern_edge > 0).sum())
    if edge_count <= 0:
        raise ValueError("Pattern has no edges for candidate generation")
    variants: List[Dict[str, float]] = []
    for scale in scales:
        win_w = int(round(w * scale))
        win_h = int(round(h * scale))
        if win_w <= 1 or win_h <= 1:
            continue
        scaled_edge_count = edge_count * (scale ** 2)
        for rot in rotations:
            variants.append(
                {
                    "scale": float(scale),
                    "rotation": float(rot),
                    "width": float(win_w),
                    "height": float(win_h),
                    "edge_count": float(scaled_edge_count),
                }
            )
    return variants


def generate_candidates_by_density(
    drawing_edge: np.ndarray,
    pattern_edge: np.ndarray,
    config: Dict[str, object],
) -> List[CandidateBox]:
    """Generate candidates using integral-image density pruning."""
    if drawing_edge is None or drawing_edge.size == 0:
        raise ValueError("Empty drawing edge provided to generate_candidates_by_density")
    if pattern_edge is None or pattern_edge.size == 0:
        raise ValueError("Empty pattern edge provided to generate_candidates_by_density")

    scales = list(config.get("scales", [1.0]))
    rotations = list(config.get("rotations", [0.0]))
    min_ratio = float(config.get("density_ratio_min", 0.4))
    max_ratio = float(config.get("density_ratio_max", 2.5))
    max_candidates = int(config.get("max_candidates", 2000))
    min_window_size = int(config.get("min_window_size", 4))

    pattern_edge_count = float((pattern_edge > 0).sum())
    if pattern_edge_count <= 0:
        raise ValueError("Pattern has no edges for candidate generation")

    h, w = drawing_edge.shape[:2]
    integral = build_integral_image(drawing_edge)

    candidates: List[CandidateBox] = []
    variants = generate_scale_rotation_variants(pattern_edge, scales, rotations)

    for var in variants:
        win_w = int(round(var["width"]))
        win_h = int(round(var["height"]))
        if win_w < min_window_size or win_h < min_window_size:
            continue
        if win_w >= w or win_h >= h:
            continue

        step = max(2, min(win_w, win_h) // 8)
        for y in range(0, h - win_h + 1, step):
            for x in range(0, w - win_w + 1, step):
                edge_count = window_sum(integral, x, y, win_w, win_h)
                if not is_density_valid(edge_count, pattern_edge_count, min_ratio, max_ratio):
                    continue
                score = density_score(edge_count, pattern_edge_count)
                candidates.append(
                    CandidateBox(
                        x=x,
                        y=y,
                        w=win_w,
                        h=win_h,
                        scale=float(var["scale"]),
                        rotation=float(var["rotation"]),
                        density_score=score,
                        raw_score=float(edge_count),
                    )
                )

    candidates.sort(key=lambda c: (c.density_score, c.raw_score), reverse=True)
    if len(candidates) > max_candidates:
        return candidates[:max_candidates]
    return candidates


@dataclass(frozen=True)
class Candidate:
    """Candidate bounding box with scale and rotation metadata."""

    x: int
    y: int
    w: int
    h: int
    scale: float
    rotation: float


def generate_candidates(
    image_shape: Tuple[int, int],
    query_shape: Tuple[int, int],
    scales: Iterable[float],
    rotations: Iterable[float],
    max_candidates: int,
) -> List[Candidate]:
    """Legacy candidate generator for compatibility."""
    h, w = image_shape
    qh, qw = query_shape
    candidates: List[Candidate] = []
    for scale in scales:
        win_w = max(4, int(round(qw * scale)))
        win_h = max(4, int(round(qh * scale)))
        if win_w >= w or win_h >= h:
            continue
        step = max(4, min(win_w, win_h) // 4)
        for y in range(0, h - win_h + 1, step):
            for x in range(0, w - win_w + 1, step):
                for rot in rotations:
                    candidates.append(Candidate(x, y, win_w, win_h, scale, rot))
                    if len(candidates) >= max_candidates:
                        return candidates
    return candidates
