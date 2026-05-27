from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

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
    """Candidate window with matching metadata."""

    x: int
    y: int
    w: int
    h: int
    scale: float
    rotation: float
    template_score: float = 0.0
    density_score: float = 0.0
    raw_score: float = 0.0


@dataclass(frozen=True)
class Candidate:
    """Legacy candidate bounding box with scale and rotation metadata."""

    x: int
    y: int
    w: int
    h: int
    scale: float
    rotation: float


def _get_float(config: Mapping[str, Any], key: str, default: float) -> float:
    """Read a float from config safely."""
    try:
        return float(config.get(key, default))
    except Exception:
        return float(default)


def _get_int(config: Mapping[str, Any], key: str, default: int) -> int:
    """Read an int from config safely."""
    try:
        return int(config.get(key, default))
    except Exception:
        return int(default)


def _get_bool(config: Mapping[str, Any], key: str, default: bool) -> bool:
    """Read a bool from config safely."""
    try:
        value = config.get(key, default)
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "y"}
        return bool(value)
    except Exception:
        return bool(default)


def _get_list_float(config: Mapping[str, Any], key: str, default: Sequence[float]) -> List[float]:
    """Read a list of floats from config safely."""
    value = config.get(key, default)
    try:
        if isinstance(value, (list, tuple, np.ndarray)):
            return [float(v) for v in value]
        return [float(value)]
    except Exception:
        return [float(v) for v in default]


def _as_binary_edge(edge: np.ndarray) -> np.ndarray:
    """Convert an edge-like image to uint8 binary map with values {0, 255}."""
    if edge is None or edge.size == 0:
        raise ValueError("Empty edge map provided.")
    arr = np.asarray(edge)
    if arr.ndim == 3:
        arr = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
    return np.where(arr > 0, 255, 0).astype(np.uint8)


def _rotate_keep_bounds(edge: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rotate an edge map while keeping complete rotated bounds."""
    if abs(float(angle_deg)) < 1e-6:
        return edge

    h, w = edge.shape[:2]
    center = (w / 2.0, h / 2.0)
    matrix = cv2.getRotationMatrix2D(center, float(angle_deg), 1.0)

    cos_value = abs(matrix[0, 0])
    sin_value = abs(matrix[0, 1])
    new_w = int(round((h * sin_value) + (w * cos_value)))
    new_h = int(round((h * cos_value) + (w * sin_value)))

    matrix[0, 2] += (new_w / 2.0) - center[0]
    matrix[1, 2] += (new_h / 2.0) - center[1]

    rotated = cv2.warpAffine(
        edge,
        matrix,
        (new_w, new_h),
        flags=cv2.INTER_NEAREST,
        borderValue=0,
    )
    return _as_binary_edge(rotated)


def _local_peak_mask(response: np.ndarray, kernel_size: int) -> np.ndarray:
    """Return mask of local maxima in a response map."""
    if response is None or response.size == 0:
        return np.zeros((0, 0), dtype=bool)
    k = max(3, int(kernel_size))
    if k % 2 == 0:
        k += 1
    kernel = np.ones((k, k), dtype=np.uint8)
    max_filtered = cv2.dilate(response.astype(np.float32), kernel)
    return response >= max_filtered


def _make_template_variant(pattern: np.ndarray, scale: float, rotation: float) -> np.ndarray:
    """Resize and rotate a pattern edge map."""
    h, w = pattern.shape[:2]
    scaled_w = int(round(w * float(scale)))
    scaled_h = int(round(h * float(scale)))
    if scaled_w <= 2 or scaled_h <= 2:
        return np.zeros((0, 0), dtype=np.uint8)
    resized = cv2.resize(pattern, (scaled_w, scaled_h), interpolation=cv2.INTER_NEAREST)
    return _rotate_keep_bounds(resized, float(rotation))


def _append_top_peaks(
    candidates: List[CandidateBox],
    response: np.ndarray,
    threshold: float,
    top_k: int,
    template_w: int,
    template_h: int,
    scale: float,
    rotation: float,
) -> None:
    """Append local peak candidates from a response map."""
    if response.size == 0:
        return
    peak_kernel = max(5, int(min(template_w, template_h) // 4))
    peak_mask = _local_peak_mask(response, peak_kernel) & (response >= float(threshold))
    if not np.any(peak_mask):
        return

    ys, xs = np.where(peak_mask)
    scores = response[ys, xs]
    if scores.size == 0:
        return

    top_k = max(1, int(top_k))
    if scores.size > top_k:
        idx = np.argpartition(-scores, top_k - 1)[:top_k]
        ys = ys[idx]
        xs = xs[idx]
        scores = scores[idx]

    for x, y, score in zip(xs.tolist(), ys.tolist(), scores.tolist()):
        candidates.append(
            CandidateBox(
                x=int(x),
                y=int(y),
                w=int(template_w),
                h=int(template_h),
                scale=float(scale),
                rotation=float(rotation),
                template_score=float(score),
                density_score=0.0,
                raw_score=float(score),
            )
        )


def generate_balanced_template_candidates(
    drawing_edge: np.ndarray,
    pattern_edge: np.ndarray,
    config: Mapping[str, Any],
) -> List[CandidateBox]:
    """Generate candidates using balanced edge template matching.

    The positive response rewards pattern-edge overlap. The background response
    softly penalizes drawing edges that appear in the template background, which
    reduces false positives in text/table regions without killing wire-connected
    symbols.
    """
    if drawing_edge is None or drawing_edge.size == 0:
        raise ValueError("Empty drawing edge provided to template matching.")
    if pattern_edge is None or pattern_edge.size == 0:
        raise ValueError("Empty pattern edge provided to template matching.")

    drawing = _as_binary_edge(drawing_edge).astype(np.float32) / 255.0
    pattern = _as_binary_edge(pattern_edge)

    drawing_h, drawing_w = drawing.shape[:2]
    scales = _get_list_float(config, "scales", [0.75, 0.9, 1.0, 1.1, 1.25])
    rotations = _get_list_float(config, "rotations", [0.0])
    max_candidates = _get_int(config, "max_candidates", 300)
    threshold = _get_float(config, "template_match_threshold", 0.12)
    top_k = _get_int(config, "template_top_k_per_variant", 100)
    background_tau = _get_float(config, "background_penalty_tau", 0.18)
    background_weight = _get_float(config, "background_penalty_weight", 0.30)
    enable_fallback = _get_bool(config, "enable_template_fallback", True)

    candidates: List[CandidateBox] = []
    fallback_candidates: List[CandidateBox] = []

    for scale in scales:
        for rotation in rotations:
            try:
                template_u8 = _make_template_variant(pattern, scale, rotation)
                if template_u8.size == 0:
                    continue
                template_h, template_w = template_u8.shape[:2]
                if template_h > drawing_h or template_w > drawing_w:
                    continue
                if template_h <= 2 or template_w <= 2:
                    continue

                template = template_u8.astype(np.float32) / 255.0
                if float(template.sum()) <= 1e-6:
                    continue

                positive = cv2.matchTemplate(drawing, template, cv2.TM_CCORR_NORMED)
                positive = np.nan_to_num(positive, nan=0.0, posinf=0.0, neginf=0.0)
                positive = np.clip(positive, 0.0, 1.0)

                fg_kernel_size = max(3, int(round(min(template_h, template_w) * 0.08)))
                if fg_kernel_size % 2 == 0:
                    fg_kernel_size += 1
                fg_kernel = np.ones((fg_kernel_size, fg_kernel_size), dtype=np.uint8)
                foreground_band = cv2.dilate(template_u8, fg_kernel, iterations=1) > 0
                background_mask = np.where(foreground_band, 0.0, 1.0).astype(np.float32)
                background_sum = float(background_mask.sum())

                if background_sum > 1.0:
                    bg_edges = cv2.matchTemplate(drawing, background_mask, cv2.TM_CCORR)
                    bg_density = bg_edges / max(background_sum, 1e-6)
                    background_clean = np.exp(-bg_density / max(background_tau, 1e-6))
                    background_clean = np.clip(background_clean, 0.0, 1.0)
                else:
                    background_clean = np.ones_like(positive, dtype=np.float32)

                wgt = float(np.clip(background_weight, 0.0, 1.0))
                balanced = positive * ((1.0 - wgt) + wgt * background_clean)
                balanced = np.nan_to_num(balanced, nan=0.0, posinf=0.0, neginf=0.0)

                _append_top_peaks(
                    candidates,
                    balanced,
                    threshold,
                    top_k,
                    template_w,
                    template_h,
                    scale,
                    rotation,
                )

                if enable_fallback:
                    _append_top_peaks(
                        fallback_candidates,
                        positive,
                        max(0.05, threshold * 0.5),
                        max(10, min(top_k, 80)),
                        template_w,
                        template_h,
                        scale,
                        rotation,
                    )
            except Exception:
                continue

    if not candidates and fallback_candidates:
        candidates = fallback_candidates

    candidates.sort(key=lambda c: c.template_score, reverse=True)
    return candidates[:max_candidates]


def generate_template_matching_candidates(
    drawing_edge: np.ndarray,
    pattern_edge: np.ndarray,
    config: Mapping[str, Any],
) -> List[CandidateBox]:
    """Compatibility wrapper for template matching candidates."""
    cfg = dict(config)
    cfg.setdefault("background_penalty_weight", 0.0)
    return generate_balanced_template_candidates(drawing_edge, pattern_edge, cfg)


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
        win_w = int(round(w * float(scale)))
        win_h = int(round(h * float(scale)))
        if win_w <= 1 or win_h <= 1:
            continue
        scaled_edge_count = edge_count * (float(scale) ** 2)
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
    config: Mapping[str, Any],
) -> List[CandidateBox]:
    """Generate candidates using integral-image density pruning."""
    if drawing_edge is None or drawing_edge.size == 0:
        raise ValueError("Empty drawing edge provided to generate_candidates_by_density")
    if pattern_edge is None or pattern_edge.size == 0:
        raise ValueError("Empty pattern edge provided to generate_candidates_by_density")

    scales = _get_list_float(config, "scales", [1.0])
    rotations = _get_list_float(config, "rotations", [0.0])
    min_ratio = _get_float(config, "density_ratio_min", 0.4)
    max_ratio = _get_float(config, "density_ratio_max", 2.5)
    max_candidates = _get_int(config, "max_candidates", 2000)
    min_window_size = _get_int(config, "min_window_size", 4)

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
                try:
                    edge_count = window_sum(integral, x, y, win_w, win_h)
                    if not is_density_valid(edge_count, pattern_edge_count, min_ratio, max_ratio):
                        continue
                    score = density_score(edge_count, pattern_edge_count)
                    candidates.append(
                        CandidateBox(
                            x=int(x),
                            y=int(y),
                            w=int(win_w),
                            h=int(win_h),
                            scale=float(var["scale"]),
                            rotation=float(var["rotation"]),
                            template_score=0.0,
                            density_score=float(score),
                            raw_score=float(edge_count),
                        )
                    )
                except Exception:
                    continue

    candidates.sort(key=lambda c: (c.density_score, c.raw_score), reverse=True)
    return candidates[:max_candidates]


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
        win_w = max(4, int(round(qw * float(scale))))
        win_h = max(4, int(round(qh * float(scale))))
        if win_w >= w or win_h >= h:
            continue
        step = max(4, min(win_w, win_h) // 4)
        for y in range(0, h - win_h + 1, step):
            for x in range(0, w - win_w + 1, step):
                for rot in rotations:
                    candidates.append(Candidate(x, y, win_w, win_h, float(scale), float(rot)))
                    if len(candidates) >= max_candidates:
                        return candidates
    return candidates
