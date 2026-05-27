from __future__ import annotations

from typing import Dict, Tuple

import cv2
import numpy as np


def _clamp01(value: float) -> float:
    """Clamp a numeric value to [0, 1]."""
    return float(np.clip(value, 0.0, 1.0))


def _as_binary_edge(edge: np.ndarray) -> np.ndarray:
    """Convert an edge-like image to uint8 binary map with values {0, 255}.

    Args:
        edge: Input edge/binary image.

    Returns:
        Binary uint8 edge map.

    Raises:
        ValueError: If the input is empty.
    """
    if edge is None or edge.size == 0:
        raise ValueError("Empty edge map provided.")

    arr = np.asarray(edge)
    if arr.ndim == 3:
        arr = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)

    return np.where(arr > 0, 255, 0).astype(np.uint8)


def _resize_to_shape(edge: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
    """Resize an edge map to target shape using nearest-neighbor interpolation."""
    target_h, target_w = target_shape
    if edge.shape[:2] == (target_h, target_w):
        return edge
    return cv2.resize(edge, (target_w, target_h), interpolation=cv2.INTER_NEAREST)


def edge_iou_score(pattern_edge: np.ndarray, candidate_edge: np.ndarray) -> float:
    """Compute edge IoU between two binary edge maps.

    Args:
        pattern_edge: Pattern edge map.
        candidate_edge: Candidate edge map.

    Returns:
        IoU score in [0, 1].
    """
    try:
        p = _as_binary_edge(pattern_edge)
        c = _as_binary_edge(candidate_edge)
        p = _resize_to_shape(p, c.shape[:2])

        p_mask = p > 0
        c_mask = c > 0
        intersection = np.logical_and(p_mask, c_mask).sum()
        union = np.logical_or(p_mask, c_mask).sum()
        if union == 0:
            return 0.0
        return _clamp01(float(intersection) / float(union))
    except Exception:
        return 0.0


def density_consistency_score(candidate_edge_count: float, pattern_edge_count: float) -> float:
    """Score consistency between candidate and pattern edge counts.

    Args:
        candidate_edge_count: Number of edge pixels in candidate.
        pattern_edge_count: Number of edge pixels in pattern.

    Returns:
        Density consistency score in [0, 1].
    """
    try:
        if pattern_edge_count <= 0 or candidate_edge_count < 0:
            return 0.0
        ratio = candidate_edge_count / max(pattern_edge_count, 1e-6)
        return _clamp01(float(np.exp(-abs(np.log(ratio + 1e-6)))))
    except Exception:
        return 0.0


def aspect_ratio_score(candidate_w: int, candidate_h: int, pattern_w: int, pattern_h: int) -> float:
    """Score aspect-ratio similarity between candidate and pattern.

    Args:
        candidate_w: Candidate width.
        candidate_h: Candidate height.
        pattern_w: Pattern width.
        pattern_h: Pattern height.

    Returns:
        Aspect-ratio score in [0, 1].
    """
    try:
        if candidate_w <= 0 or candidate_h <= 0 or pattern_w <= 0 or pattern_h <= 0:
            return 0.0
        candidate_ar = float(candidate_w) / float(candidate_h)
        pattern_ar = float(pattern_w) / float(pattern_h)
        return _clamp01(
            float(np.exp(-abs(np.log((candidate_ar + 1e-6) / (pattern_ar + 1e-6)))))
        )
    except Exception:
        return 0.0


def edge_coverage_score(source_edge: np.ndarray, target_edge: np.ndarray, tolerance: int = 2) -> float:
    """Measure how much of source edges are covered by target edges.

    Args:
        source_edge: Source edge map.
        target_edge: Target edge map.
        tolerance: Pixel tolerance.

    Returns:
        Coverage score in [0, 1].
    """
    try:
        source = _as_binary_edge(source_edge)
        target = _as_binary_edge(target_edge)
        target = _resize_to_shape(target, source.shape[:2])

        source_mask = source > 0
        if int(source_mask.sum()) == 0:
            return 0.0

        dt_input = np.where(target > 0, 0, 255).astype(np.uint8)
        distance = cv2.distanceTransform(dt_input, cv2.DIST_L2, 3)
        covered = distance[source_mask] <= float(max(0, tolerance))
        return _clamp01(float(covered.mean()))
    except Exception:
        return 0.0


def bidirectional_edge_f1(
    pattern_edge: np.ndarray,
    candidate_edge: np.ndarray,
    tolerance: int = 2,
) -> Tuple[float, float, float]:
    """Compute candidate-to-pattern precision, pattern-to-candidate recall and F1.

    Args:
        pattern_edge: Pattern edge map.
        candidate_edge: Candidate edge map.
        tolerance: Pixel tolerance.

    Returns:
        Tuple of (precision, recall, f1), all in [0, 1].
    """
    try:
        pattern = _as_binary_edge(pattern_edge)
        candidate = _as_binary_edge(candidate_edge)
        pattern = _resize_to_shape(pattern, candidate.shape[:2])

        precision = edge_coverage_score(candidate, pattern, tolerance=tolerance)
        recall = edge_coverage_score(pattern, candidate, tolerance=tolerance)
        f1 = (2.0 * precision * recall) / max(precision + recall, 1e-6)
        return _clamp01(precision), _clamp01(recall), _clamp01(f1)
    except Exception:
        return 0.0, 0.0, 0.0


def masked_edge_precision_score(
    pattern_edge: np.ndarray,
    candidate_edge: np.ndarray,
    dilation: int = 2,
) -> float:
    """Measure candidate edge pixels explained by a dilated pattern mask.

    This penalizes text/table candidates that contain many extra strokes outside
    the query shape.

    Args:
        pattern_edge: Pattern edge map.
        candidate_edge: Candidate edge map.
        dilation: Dilation radius for the allowed pattern mask.

    Returns:
        Masked precision score in [0, 1].
    """
    try:
        pattern = _as_binary_edge(pattern_edge)
        candidate = _as_binary_edge(candidate_edge)
        pattern = _resize_to_shape(pattern, candidate.shape[:2])

        k = max(1, int(dilation) * 2 + 1)
        kernel = np.ones((k, k), dtype=np.uint8)
        allowed = cv2.dilate(pattern, kernel, iterations=1) > 0
        cand_mask = candidate > 0
        total = int(cand_mask.sum())
        if total <= 0:
            return 0.0
        inside = int(np.logical_and(cand_mask, allowed).sum())
        return _clamp01(float(inside) / float(total))
    except Exception:
        return 0.0


def outside_edge_ratio_score(
    pattern_edge: np.ndarray,
    candidate_edge: np.ndarray,
    dilation: int = 2,
    tau: float = 2.5,
) -> Tuple[float, float]:
    """Score candidates by penalizing edges outside the dilated pattern mask.

    Args:
        pattern_edge: Pattern edge map.
        candidate_edge: Candidate edge map.
        dilation: Dilation radius for allowed pattern mask.
        tau: Exponential decay factor.

    Returns:
        Tuple of (score, outside_ratio).
    """
    try:
        pattern = _as_binary_edge(pattern_edge)
        candidate = _as_binary_edge(candidate_edge)
        pattern = _resize_to_shape(pattern, candidate.shape[:2])

        k = max(1, int(dilation) * 2 + 1)
        kernel = np.ones((k, k), dtype=np.uint8)
        allowed = cv2.dilate(pattern, kernel, iterations=1) > 0
        cand_mask = candidate > 0

        outside_edges = int(np.logical_and(cand_mask, np.logical_not(allowed)).sum())
        pattern_edges = int((pattern > 0).sum())
        outside_ratio = float(outside_edges) / max(float(pattern_edges), 1e-6)
        score = float(np.exp(-outside_ratio / max(float(tau), 1e-6)))
        return _clamp01(score), float(outside_ratio)
    except Exception:
        return 0.0, float("inf")


def candidate_shape_cleanliness_score(
    pattern_edge: np.ndarray,
    candidate_edge: np.ndarray,
    dilation: int = 2,
) -> float:
    """Combine masked precision and outside-edge penalty.

    Args:
        pattern_edge: Pattern edge map.
        candidate_edge: Candidate edge map.
        dilation: Pattern mask dilation radius.

    Returns:
        Shape cleanliness score in [0, 1].
    """
    try:
        masked = masked_edge_precision_score(pattern_edge, candidate_edge, dilation=dilation)
        outside_score, _ = outside_edge_ratio_score(pattern_edge, candidate_edge, dilation=dilation)
        return _clamp01(0.65 * masked + 0.35 * outside_score)
    except Exception:
        return 0.0


def count_small_components(binary_or_edge: np.ndarray, min_area: int = 3) -> int:
    """Count connected components whose area is at least min_area."""
    try:
        edge = _as_binary_edge(binary_or_edge)
        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(edge, connectivity=8)
        if num_labels <= 1:
            return 0
        areas = stats[1:, cv2.CC_STAT_AREA]
        return int((areas >= int(min_area)).sum())
    except Exception:
        return 0


def edge_density_score(edge: np.ndarray, max_density: float = 0.20) -> float:
    """Return cleanliness score based on edge density.

    Args:
        edge: Edge map.
        max_density: Density above which the block becomes suspicious.

    Returns:
        Score in [0, 1], where 1 is sparse/clean.
    """
    try:
        binary = _as_binary_edge(edge)
        density = float((binary > 0).mean())
        return _clamp01(1.0 - min(1.0, density / max(max_density, 1e-6)))
    except Exception:
        return 0.0


def line_grid_score(edge: np.ndarray) -> float:
    """Return cleanliness score based on horizontal/vertical grid evidence.

    Args:
        edge: Edge map.

    Returns:
        Score in [0, 1], where lower means more table/grid-like.
    """
    try:
        binary = _as_binary_edge(edge)
        h, w = binary.shape[:2]
        if h <= 1 or w <= 1:
            return 0.0
        mask = binary > 0
        row_density = mask.sum(axis=1) / float(w)
        col_density = mask.sum(axis=0) / float(h)
        horizontal_line_ratio = float((row_density > 0.35).mean())
        vertical_line_ratio = float((col_density > 0.35).mean())
        line_ratio = min(1.0, horizontal_line_ratio + vertical_line_ratio)
        return _clamp01(float(np.exp(-3.0 * line_ratio)))
    except Exception:
        return 0.0


def component_clutter_score(
    edge: np.ndarray,
    max_components: int = 25,
    min_area: int = 3,
) -> float:
    """Return cleanliness score based on connected-component clutter."""
    try:
        components = count_small_components(edge, min_area=min_area)
        if max_components <= 0:
            return 1.0 if components == 0 else 0.0
        return _clamp01(float(np.exp(-max(0, components - max_components) / float(max_components))))
    except Exception:
        return 0.0


def artifact_penalty_score(edge: np.ndarray) -> float:
    """Estimate whether a block is clean/symbol-like or artifact-like.

    Args:
        edge: Edge map block.

    Returns:
        Cleanliness score in [0, 1]. 1 means clean, 0 means table/text-like.
    """
    try:
        density_clean = edge_density_score(edge, max_density=0.20)
        grid_clean = line_grid_score(edge)
        comp_clean = component_clutter_score(edge, max_components=25, min_area=3)
        return _clamp01(0.40 * density_clean + 0.30 * grid_clean + 0.30 * comp_clean)
    except Exception:
        return 0.0


def build_artifact_mask(
    edge: np.ndarray,
    block_size: int = 64,
    stride: int = 32,
    threshold: float = 0.35,
    dilate_kernel: int = 25,
) -> np.ndarray:
    """Build a table/text artifact mask from a drawing edge map.

    Convention:
        255 = artifact-heavy region, 0 = clean region.

    Args:
        edge: Drawing edge map.
        block_size: Coarse analysis block size.
        stride: Coarse analysis stride.
        threshold: Artifact score threshold. Lower values mark more regions.
        dilate_kernel: Dilation kernel size for mask expansion.

    Returns:
        Artifact mask with same height/width as edge.
    """
    try:
        binary = _as_binary_edge(edge)
        h, w = binary.shape[:2]
        block_size = max(16, int(block_size))
        stride = max(8, int(stride))
        threshold = float(threshold)

        mask = np.zeros((h, w), dtype=np.uint8)
        if h == 0 or w == 0:
            return mask

        for y in range(0, h, stride):
            y2 = min(h, y + block_size)
            y1 = max(0, y2 - block_size)
            for x in range(0, w, stride):
                x2 = min(w, x + block_size)
                x1 = max(0, x2 - block_size)
                block = binary[y1:y2, x1:x2]
                if block.size == 0:
                    continue

                edge_density = float((block > 0).mean())
                if edge_density < 0.015:
                    continue

                block_h, block_w = block.shape[:2]
                block_bool = block > 0
                row_density = block_bool.sum(axis=1) / float(max(block_w, 1))
                col_density = block_bool.sum(axis=0) / float(max(block_h, 1))
                horizontal_line_ratio = float((row_density > 0.35).mean())
                vertical_line_ratio = float((col_density > 0.35).mean())
                line_ratio = min(1.0, horizontal_line_ratio + vertical_line_ratio)

                components = count_small_components(block, min_area=3)
                density_norm = min(1.0, edge_density / 0.18)
                component_norm = min(1.0, components / 25.0)

                artifact_score = (
                    0.40 * density_norm
                    + 0.30 * component_norm
                    + 0.30 * line_ratio
                )

                if artifact_score >= threshold:
                    mask[y1:y2, x1:x2] = 255

        k = int(dilate_kernel)
        if k > 1:
            if k % 2 == 0:
                k += 1
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
            mask = cv2.dilate(mask, kernel, iterations=1)

        return mask.astype(np.uint8)
    except Exception:
        if edge is None:
            return np.zeros((0, 0), dtype=np.uint8)
        return np.zeros(edge.shape[:2], dtype=np.uint8)


def box_artifact_penalty(
    artifact_mask: np.ndarray,
    x: int,
    y: int,
    w: int,
    h: int,
    floor: float = 0.15,
) -> float:
    """Compute candidate penalty from overlap with artifact mask.

    Args:
        artifact_mask: Mask where 255 indicates artifact-heavy region.
        x: Box x.
        y: Box y.
        w: Box width.
        h: Box height.
        floor: Minimum returned penalty.

    Returns:
        Penalty in [floor, 1]. Higher is cleaner.
    """
    try:
        if artifact_mask is None or artifact_mask.size == 0 or w <= 0 or h <= 0:
            return 1.0
        mask_h, mask_w = artifact_mask.shape[:2]
        x1 = max(0, int(x))
        y1 = max(0, int(y))
        x2 = min(mask_w, int(x + w))
        y2 = min(mask_h, int(y + h))
        if x2 <= x1 or y2 <= y1:
            return 1.0
        roi = artifact_mask[y1:y2, x1:x2]
        artifact_overlap = float((roi > 0).mean()) if roi.size else 0.0
        return _clamp01(max(float(floor), 1.0 - artifact_overlap))
    except Exception:
        return 1.0


def build_search_region_mask(
    edge: np.ndarray,
    block_size: int = 96,
    stride: int = 48,
    max_artifact_score: float = 0.55,
) -> np.ndarray:
    """Build an allowed-search mask where 255 means allowed and 0 means artifact."""
    artifact_mask = build_artifact_mask(
        edge=edge,
        block_size=block_size,
        stride=stride,
        threshold=max_artifact_score,
        dilate_kernel=max(15, block_size // 3),
    )
    return np.where(artifact_mask > 0, 0, 255).astype(np.uint8)


def box_mask_coverage(mask: np.ndarray, x: int, y: int, w: int, h: int) -> float:
    """Compute the fraction of a candidate box covered by a positive mask."""
    try:
        if mask is None or mask.size == 0 or w <= 0 or h <= 0:
            return 1.0
        mask_h, mask_w = mask.shape[:2]
        x1 = max(0, int(x))
        y1 = max(0, int(y))
        x2 = min(mask_w, int(x + w))
        y2 = min(mask_h, int(y + h))
        if x2 <= x1 or y2 <= y1:
            return 0.0
        roi = mask[y1:y2, x1:x2]
        return _clamp01(float((roi > 0).mean()))
    except Exception:
        return 1.0


def analyze_pattern_connectivity(
    pattern_edge: np.ndarray,
    band_ratio: float = 0.12,
    connector_width_ratio: float = 0.22,
    connector_touch_threshold: float = 0.015,
) -> Dict[str, float | bool]:
    """Analyze whether a query pattern is embedded near opposite borders.

    The method inspects center corridors on each border, avoiding false
    positives from closed shapes that touch the crop frame.

    Args:
        pattern_edge: Cropped pattern edge map.
        band_ratio: Relative width of the near-border band.
        connector_width_ratio: Width of the center corridor relative to size.
        connector_touch_threshold: Minimum touch ratio for embedded decision.

    Returns:
        Dictionary containing touch ratios and embedded flags.
    """
    try:
        edge = _as_binary_edge(pattern_edge)
        h, w = edge.shape[:2]
        fg = edge > 0
        total = float(fg.sum())
        if h <= 2 or w <= 2 or total <= 0:
            return {
                "total_edge_pixels": 0.0,
                "left_center_touch_ratio": 0.0,
                "right_center_touch_ratio": 0.0,
                "top_center_touch_ratio": 0.0,
                "bottom_center_touch_ratio": 0.0,
                "horizontal_embedded": False,
                "vertical_embedded": False,
                "is_embedded": False,
            }

        band = max(2, int(round(min(h, w) * float(band_ratio))))
        center_band_w = max(2, int(round(w * float(connector_width_ratio))))
        center_band_h = max(2, int(round(h * float(connector_width_ratio))))

        cy = h // 2
        cx = w // 2
        y1 = max(0, cy - center_band_h // 2)
        y2 = min(h, cy + center_band_h // 2 + 1)
        x1 = max(0, cx - center_band_w // 2)
        x2 = min(w, cx + center_band_w // 2 + 1)

        left_center = float(fg[y1:y2, :band].sum()) / total
        right_center = float(fg[y1:y2, w - band :].sum()) / total
        top_center = float(fg[:band, x1:x2].sum()) / total
        bottom_center = float(fg[h - band :, x1:x2].sum()) / total

        horizontal = bool(left_center > connector_touch_threshold and right_center > connector_touch_threshold)
        vertical = bool(top_center > connector_touch_threshold and bottom_center > connector_touch_threshold)

        return {
            "total_edge_pixels": float(total),
            "left_center_touch_ratio": float(left_center),
            "right_center_touch_ratio": float(right_center),
            "top_center_touch_ratio": float(top_center),
            "bottom_center_touch_ratio": float(bottom_center),
            "horizontal_embedded": horizontal,
            "vertical_embedded": vertical,
            "is_embedded": bool(horizontal or vertical),
        }
    except Exception:
        return {
            "total_edge_pixels": 0.0,
            "left_center_touch_ratio": 0.0,
            "right_center_touch_ratio": 0.0,
            "top_center_touch_ratio": 0.0,
            "bottom_center_touch_ratio": 0.0,
            "horizontal_embedded": False,
            "vertical_embedded": False,
            "is_embedded": False,
        }


def closed_shape_score(edge: np.ndarray) -> float:
    """Estimate closed/circular shape evidence in an edge map.

    Args:
        edge: Edge map.

    Returns:
        Score in [0, 1]. Higher means more closed-shape evidence.
    """
    try:
        binary = _as_binary_edge(edge)
        h, w = binary.shape[:2]
        if h <= 2 or w <= 2:
            return 0.0

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)

        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return 0.0

        area_threshold = max(4.0, 0.004 * float(h * w))
        best = 0.0
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < area_threshold:
                continue
            perimeter = float(cv2.arcLength(contour, True))
            if perimeter <= 1e-6:
                continue
            circularity = (4.0 * float(np.pi) * area) / max(perimeter * perimeter, 1e-6)
            x, y, cw, ch = cv2.boundingRect(contour)
            rect_area = float(cw * ch)
            rectangularity = area / max(rect_area, 1e-6)
            hull = cv2.convexHull(contour)
            hull_area = float(cv2.contourArea(hull))
            solidity = area / max(hull_area, 1e-6)
            closedness = max(circularity, rectangularity * solidity)
            best = max(best, closedness)

        return _clamp01(best)
    except Exception:
        return 0.0


def candidate_closed_shape_score(candidate_edge: np.ndarray) -> float:
    """Wrapper for closed-shape evidence on candidates."""
    return closed_shape_score(candidate_edge)


def closed_shape_compatibility_score(
    query_edge: np.ndarray,
    candidate_edge: np.ndarray,
) -> float:
    """Return compatibility between query and candidate closed-shape evidence."""
    try:
        query_closed = closed_shape_score(query_edge)
        candidate_closed = closed_shape_score(candidate_edge)
        if query_closed < 0.35:
            return 1.0
        score = float(np.exp(-abs(query_closed - candidate_closed) / 0.35))
        return _clamp01(score)
    except Exception:
        return 0.0


def connection_aware_outside_edge_ratio_score(
    pattern_edge: np.ndarray,
    candidate_edge: np.ndarray,
    dilation: int = 2,
    tau: float = 2.5,
    connector_width_ratio: float = 0.18,
    topology: Dict[str, float | bool] | None = None,
) -> Tuple[float, float, Dict[str, float | bool]]:
    """Penalize outside edges while allowing expected connector wires.

    For embedded/open query patterns, a narrow band through the candidate center
    is considered an allowed connector region. This helps true circuit symbols
    connected to wires while still penalizing random text/table strokes.

    Args:
        pattern_edge: Pattern edge map.
        candidate_edge: Candidate edge map.
        dilation: Pattern mask dilation radius.
        tau: Exponential decay for outside ratio.
        connector_width_ratio: Width of allowed connector band relative to box.

    Returns:
        Tuple of (score, outside_ratio, topology).
    """
    try:
        pattern = _as_binary_edge(pattern_edge)
        candidate = _as_binary_edge(candidate_edge)
        pattern = _resize_to_shape(pattern, candidate.shape[:2])
        h, w = candidate.shape[:2]

        k = max(1, int(dilation) * 2 + 1)
        kernel = np.ones((k, k), dtype=np.uint8)
        allowed = cv2.dilate(pattern, kernel, iterations=1) > 0

        if topology is None:
            topology = analyze_pattern_connectivity(pattern)
        if bool(topology.get("vertical_embedded", False)):
            band_w = max(2, int(round(w * float(connector_width_ratio))))
            cx = w // 2
            x1 = max(0, cx - band_w // 2)
            x2 = min(w, cx + band_w // 2 + 1)
            allowed[:, x1:x2] = True
        if bool(topology.get("horizontal_embedded", False)):
            band_h = max(2, int(round(h * float(connector_width_ratio))))
            cy = h // 2
            y1 = max(0, cy - band_h // 2)
            y2 = min(h, cy + band_h // 2 + 1)
            allowed[y1:y2, :] = True

        cand_mask = candidate > 0
        outside_edges = int(np.logical_and(cand_mask, np.logical_not(allowed)).sum())
        pattern_edges = int((pattern > 0).sum())
        outside_ratio = float(outside_edges) / max(float(pattern_edges), 1e-6)
        score = float(np.exp(-outside_ratio / max(float(tau), 1e-6)))
        return _clamp01(score), float(outside_ratio), topology
    except Exception:
        return 0.0, float("inf"), {
            "vertical_embedded": False,
            "horizontal_embedded": False,
            "is_embedded": False,
        }


def build_layout_suppression_mask(
    edge: np.ndarray,
    block_size: int = 96,
    stride: int = 48,
    artifact_threshold: float = 0.34,
    border_band_ratio: float = 0.055,
    dilate_kernel: int = 31,
) -> np.ndarray:
    """Build a layout artifact mask for border grids, tables and notes.

    Convention: 255 means layout/artifact-heavy, 0 means clean schematic region.
    The method uses geometry only, not fixed BOM coordinates.

    Args:
        edge: Drawing edge map.
        block_size: Coarse block size.
        stride: Coarse block stride.
        artifact_threshold: Block artifact threshold.
        border_band_ratio: Relative width of frame/grid border bands.
        dilate_kernel: Dilation size to connect detected layout regions.

    Returns:
        uint8 mask with the same image size as edge.
    """
    try:
        binary = _as_binary_edge(edge)
        h, w = binary.shape[:2]
        mask = build_artifact_mask(
            binary,
            block_size=block_size,
            stride=stride,
            threshold=artifact_threshold,
            dilate_kernel=dilate_kernel,
        )

        if h <= 0 or w <= 0:
            return mask

        # Suppress border grid bands by proportional frame bands. This is still
        # coordinate-free relative to image layout and removes row/column labels.
        by = max(8, int(round(h * float(border_band_ratio))))
        bx = max(8, int(round(w * float(border_band_ratio))))

        # Only mark the border band if it contains actual edge evidence.
        if float((binary[:by, :] > 0).mean()) > 0.003:
            mask[: by + 2, :] = 255
        if float((binary[h - by :, :] > 0).mean()) > 0.003:
            mask[h - by - 2 :, :] = 255
        if float((binary[:, :bx] > 0).mean()) > 0.003:
            mask[:, : bx + 2] = 255
        if float((binary[:, w - bx :] > 0).mean()) > 0.003:
            mask[:, w - bx - 2 :] = 255

        # Strong horizontal/vertical table lines are often title blocks/legends.
        # Morphological line extraction marks connected grid-like areas.
        horizontal_len = max(25, w // 18)
        vertical_len = max(25, h // 18)
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_len, 1))
        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vertical_len))
        horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel)
        vertical = cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_kernel)
        grid = cv2.bitwise_or(horizontal, vertical)
        if int((grid > 0).sum()) > 0:
            gk = max(15, dilate_kernel)
            if gk % 2 == 0:
                gk += 1
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (gk, gk))
            grid = cv2.dilate(grid, kernel, iterations=1)
            # Do not let isolated circuit wires suppress the whole schematic; only
            # merge grid evidence with the block artifact mask and border bands.
            mask = np.where(grid > 0, np.maximum(mask, 128).astype(np.uint8), mask)

        return np.where(mask > 0, 255, 0).astype(np.uint8)
    except Exception:
        if edge is None:
            return np.zeros((0, 0), dtype=np.uint8)
        return np.zeros(edge.shape[:2], dtype=np.uint8)


def box_layout_penalty(
    layout_mask: np.ndarray,
    x: int,
    y: int,
    w: int,
    h: int,
    floor: float = 0.12,
) -> float:
    """Return a cleanliness penalty from overlap with layout suppression mask."""
    return box_artifact_penalty(layout_mask, x=x, y=y, w=w, h=h, floor=floor)


def fuse_scores(
    chamfer_score: float,
    edge_iou: float,
    density: float,
    aspect: float,
    weights: Dict[str, float] | None = None,
) -> float:
    """Fuse legacy sub-scores into a confidence value in [0, 1]."""
    try:
        if weights is None:
            weights = {"chamfer": 0.50, "edge_iou": 0.20, "density": 0.20, "aspect": 0.10}
        total_weight = sum(float(v) for v in weights.values())
        if total_weight <= 0:
            return 0.0
        score = (
            float(weights.get("chamfer", 0.0)) * chamfer_score
            + float(weights.get("edge_iou", 0.0)) * edge_iou
            + float(weights.get("density", 0.0)) * density
            + float(weights.get("aspect", 0.0)) * aspect
        ) / total_weight
        return _clamp01(float(score))
    except Exception:
        return 0.0


def edge_iou(query_edges: np.ndarray, target_edges: np.ndarray) -> float:
    """Backward-compatible wrapper for edge_iou_score."""
    return edge_iou_score(query_edges, target_edges)


def density_ratio_score(query_density: float, candidate_density: float) -> float:
    """Backward-compatible density-ratio score."""
    try:
        if query_density <= 0:
            return 0.0
        ratio = candidate_density / max(query_density, 1e-6)
        return _clamp01(float(np.exp(-abs(np.log(ratio + 1e-6)))))
    except Exception:
        return 0.0
