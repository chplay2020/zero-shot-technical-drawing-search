from __future__ import annotations

import time
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np

from pattern_detector.candidate_generator import (
    CandidateBox,
    generate_balanced_template_candidates,
    generate_candidates_by_density,
    generate_template_matching_candidates,
)
from pattern_detector.chamfer import (
    build_directional_distance_transforms,
    chamfer_distance_to_score,
    compute_gradient_orientation,
    directional_chamfer_distance,
    extract_oriented_edge_points,
    quantize_orientation,
)
from pattern_detector.nms import nms
from pattern_detector.preprocessing import preprocess_drawing, preprocess_pattern
from pattern_detector.scoring import (
    aspect_ratio_score,
    bidirectional_edge_f1,
    analyze_pattern_connectivity,
    box_artifact_penalty,
    box_layout_penalty,
    build_artifact_mask,
    build_layout_suppression_mask,
    closed_shape_score,
    candidate_shape_cleanliness_score,
    connection_aware_outside_edge_ratio_score,
    density_consistency_score,
    edge_iou_score,
    masked_edge_precision_score,
    outside_edge_ratio_score,
)
from pattern_detector.visualization import draw_detections


class PatternDetector:
    """CPU-friendly zero-shot pattern detector with budgeted candidate search.

    The detector uses a coarse-to-fine classical CV pipeline. In accurate mode,
    it merges multiple candidate sources instead of relying on a single tuned
    threshold: balanced edge template matching, plain edge template matching,
    and density fallback. The merged pool is then verified by shape scores.
    """

    def __init__(self, config: Any) -> None:
        """Initialize detector with either a dict or a simple config object.

        Args:
            config: Configuration dictionary or dataclass-like object.
        """
        if isinstance(config, dict):
            self.config = config.copy()
        elif hasattr(config, "to_dict"):
            self.config = config.to_dict()
        elif hasattr(config, "__dict__"):
            self.config = vars(config).copy()
        else:
            raise TypeError("Unsupported config type. Expected dict or config object.")


    @staticmethod
    def _rotate_keep_bounds(edge: np.ndarray, angle_deg: float) -> np.ndarray:
        """Rotate a binary edge map while preserving full rotated bounds.

        Args:
            edge: Binary edge map.
            angle_deg: Rotation angle in degrees.

        Returns:
            Rotated binary edge map.
        """
        if abs(float(angle_deg)) < 1e-6:
            return edge

        h, w = edge.shape[:2]
        center = (w / 2.0, h / 2.0)
        matrix = cv2.getRotationMatrix2D(center, float(angle_deg), 1.0)

        cos_value = abs(matrix[0, 0])
        sin_value = abs(matrix[0, 1])
        new_w = max(1, int(round((h * sin_value) + (w * cos_value))))
        new_h = max(1, int(round((h * cos_value) + (w * sin_value))))

        matrix[0, 2] += (new_w / 2.0) - center[0]
        matrix[1, 2] += (new_h / 2.0) - center[1]

        rotated = cv2.warpAffine(
            edge,
            matrix,
            (new_w, new_h),
            flags=cv2.INTER_NEAREST,
            borderValue=0,
        )
        return np.where(rotated > 0, 255, 0).astype(np.uint8)

    @classmethod
    def _make_candidate_pattern_edge(cls, pattern_edge: np.ndarray, cand: CandidateBox) -> np.ndarray:
        """Create a candidate-specific pattern edge using candidate scale/rotation.

        Template matching already searches over scale and rotation. The verifier
        must use the same transformed query; otherwise rotated matches are
        incorrectly verified with the unrotated pattern and get low scores.

        Args:
            pattern_edge: Cropped query edge map.
            cand: Candidate box containing scale and rotation metadata.

        Returns:
            Binary pattern edge map with shape (cand.h, cand.w).
        """
        base = np.where(pattern_edge > 0, 255, 0).astype(np.uint8)
        h, w = base.shape[:2]
        scaled_w = max(2, int(round(w * float(cand.scale))))
        scaled_h = max(2, int(round(h * float(cand.scale))))
        scaled = cv2.resize(base, (scaled_w, scaled_h), interpolation=cv2.INTER_NEAREST)
        rotated = cls._rotate_keep_bounds(scaled, float(cand.rotation))

        if rotated.shape[:2] != (int(cand.h), int(cand.w)):
            rotated = cv2.resize(
                rotated,
                (int(cand.w), int(cand.h)),
                interpolation=cv2.INTER_NEAREST,
            )
        return np.where(rotated > 0, 255, 0).astype(np.uint8)

    @staticmethod
    def _candidate_iou(a: CandidateBox, b: CandidateBox) -> float:
        """Compute IoU between two candidate boxes."""
        ax1, ay1, ax2, ay2 = a.x, a.y, a.x + a.w, a.y + a.h
        bx1, by1, bx2, by2 = b.x, b.y, b.x + b.w, b.y + b.h
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
        inter = float(iw * ih)
        if inter <= 0:
            return 0.0
        area_a = float(max(1, a.w * a.h))
        area_b = float(max(1, b.w * b.h))
        return inter / max(area_a + area_b - inter, 1e-6)

    @staticmethod
    def _candidate_score(candidate: CandidateBox) -> float:
        """Return a robust sort score for candidate merging."""
        return float(max(candidate.template_score, candidate.density_score, candidate.raw_score))

    def _merge_candidates(self, groups: List[List[CandidateBox]]) -> List[CandidateBox]:
        """Merge candidate sources with candidate-level NMS for diversity."""
        all_candidates: List[CandidateBox] = []
        for group in groups:
            all_candidates.extend([c for c in group if isinstance(c, CandidateBox)])

        if not all_candidates:
            return []

        all_candidates.sort(key=self._candidate_score, reverse=True)
        max_pool = int(self.config.get("candidate_pool_size", self.config.get("max_candidates", 1200)))
        merge_iou = float(self.config.get("candidate_merge_iou", 0.82))

        kept: List[CandidateBox] = []
        for cand in all_candidates:
            if all(self._candidate_iou(cand, old) < merge_iou for old in kept):
                kept.append(cand)
            if len(kept) >= max_pool:
                break
        return kept

    def _generate_candidates(self, drawing_edge: np.ndarray, pattern_edge: np.ndarray) -> List[CandidateBox]:
        """Generate a broad, diverse candidate pool.

        The previous pipeline used only balanced template matching. That is
        precise for isolated symbols but can miss embedded symbols such as
        resistors connected to long wires. This method merges multiple sources:
        balanced template candidates, plain positive template candidates, and
        density fallback candidates. Verification later decides which are valid.
        """
        groups: List[List[CandidateBox]] = []

        try:
            groups.append(
                generate_balanced_template_candidates(
                    drawing_edge=drawing_edge,
                    pattern_edge=pattern_edge,
                    config=self.config,
                )
            )
        except Exception:
            groups.append([])

        if bool(self.config.get("enable_candidate_ensemble", True)):
            try:
                plain_cfg = dict(self.config)
                plain_cfg["background_penalty_weight"] = 0.0
                plain_cfg["template_match_threshold"] = float(
                    self.config.get("plain_template_threshold", 0.08)
                )
                plain_cfg["template_top_k_per_variant"] = int(
                    self.config.get("plain_template_top_k_per_variant", 160)
                )
                plain_cfg["max_candidates"] = int(
                    self.config.get("plain_template_max_candidates", 900)
                )
                groups.append(
                    generate_template_matching_candidates(
                        drawing_edge=drawing_edge,
                        pattern_edge=pattern_edge,
                        config=plain_cfg,
                    )
                )
            except Exception:
                groups.append([])

            try:
                density_cfg = dict(self.config)
                density_cfg["max_candidates"] = int(self.config.get("density_fallback_max_candidates", 450))
                groups.append(
                    generate_candidates_by_density(
                        drawing_edge=drawing_edge,
                        pattern_edge=pattern_edge,
                        config=density_cfg,
                    )
                )
            except Exception:
                groups.append([])

        candidates = self._merge_candidates(groups)
        if candidates:
            return candidates

        return generate_candidates_by_density(
            drawing_edge=drawing_edge,
            pattern_edge=pattern_edge,
            config=self.config,
        )

    def _adaptive_filter(self, raw_detections: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Filter raw detections using strict threshold plus recall fallback.

        This is not blind parameter tuning. It first applies a strict dynamic
        threshold. If too few detections survive, it uses the already-computed
        score distribution and selects a spatially diverse top set above a safe
        minimum score. This uses the larger CPU candidate search without forcing
        a single global threshold to work for all pattern types.
        """
        if not raw_detections:
            return [], {
                "max_raw_score": 0.0,
                "dynamic_confidence_threshold": 0.0,
                "num_after_dynamic_threshold": 0,
                "adaptive_recall_used": False,
            }

        raw_sorted = sorted(raw_detections, key=lambda det: float(det.get("confidence", 0.0)), reverse=True)
        max_raw_score = float(raw_sorted[0].get("confidence", 0.0))
        dynamic_min = float(self.config.get("dynamic_min_threshold", 0.48))
        dynamic_ratio = float(self.config.get("dynamic_threshold_ratio", 0.88))
        dynamic_threshold = max(dynamic_min, max_raw_score * dynamic_ratio)

        filtered = [det for det in raw_sorted if float(det.get("confidence", 0.0)) >= dynamic_threshold]
        num_after_dynamic = len(filtered)
        adaptive_used = False

        min_final = int(self.config.get("min_final_candidates_before_nms", 6))
        if len(filtered) < min_final and bool(self.config.get("enable_adaptive_recall", True)):
            recall_ratio = float(self.config.get("adaptive_recall_ratio", 0.74))
            recall_floor = float(self.config.get("adaptive_recall_min_score", 0.36))
            recall_threshold = max(recall_floor, max_raw_score * recall_ratio)
            recall_candidates = [
                det for det in raw_sorted if float(det.get("confidence", 0.0)) >= recall_threshold
            ]
            if len(recall_candidates) > len(filtered):
                filtered = recall_candidates
                adaptive_used = True

        if not filtered and bool(self.config.get("enable_empty_fallback", True)):
            fallback_min = float(self.config.get("fallback_min_score", 0.20))
            fallback_top_k = int(self.config.get("fallback_top_k", 8))
            filtered = [det for det in raw_sorted[:fallback_top_k] if float(det.get("confidence", 0.0)) >= fallback_min]
            adaptive_used = bool(filtered)

        pre_nms_top_k = int(self.config.get("pre_nms_top_k", 24))
        filtered = filtered[: max(1, pre_nms_top_k)]

        return filtered, {
            "max_raw_score": max_raw_score,
            "dynamic_confidence_threshold": dynamic_threshold,
            "num_after_dynamic_threshold": num_after_dynamic,
            "num_after_topk_filter": len(filtered),
            "adaptive_recall_used": adaptive_used,
            "pre_nms_top_k": pre_nms_top_k,
        }

    def detect(
        self,
        pattern_image: np.ndarray,
        drawing_image: np.ndarray,
    ) -> Tuple[List[Dict[str, Any]], np.ndarray, Dict[str, Any]]:
        """Run full detection pipeline.

        Args:
            pattern_image: Query pattern image.
            drawing_image: Large technical drawing image.

        Returns:
            Tuple of detections, visualization image, and metadata.
        """
        start = time.time()

        if pattern_image is None or pattern_image.size == 0:
            raise ValueError("Empty pattern image provided to detect.")
        if drawing_image is None or drawing_image.size == 0:
            raise ValueError("Empty drawing image provided to detect.")

        pattern_data = preprocess_pattern(pattern_image, self.config)
        drawing_data = preprocess_drawing(drawing_image, self.config)

        pattern_edge = pattern_data["cropped_edge"]
        pattern_h, pattern_w = pattern_edge.shape[:2]

        drawing_edge = drawing_data["edge"]
        drawing_gray = drawing_data["gray"]
        drawing_resized = drawing_data["resized"]
        scale_factor = float(drawing_data["scale_factor"])

        candidates = self._generate_candidates(drawing_edge, pattern_edge)
        max_verify = int(self.config.get("max_verification_candidates", len(candidates)))
        candidates = candidates[: max(1, max_verify)]

        artifact_mask = build_artifact_mask(
            drawing_edge,
            block_size=int(self.config.get("artifact_block_size", 64)),
            stride=int(self.config.get("artifact_stride", 32)),
            threshold=float(self.config.get("artifact_threshold", 0.35)),
            dilate_kernel=int(self.config.get("artifact_dilate_kernel", 25)),
        )
        artifact_mask_coverage = float((artifact_mask > 0).mean()) if artifact_mask.size else 0.0

        layout_mask = build_layout_suppression_mask(
            drawing_edge,
            block_size=int(self.config.get("layout_block_size", 96)),
            stride=int(self.config.get("layout_stride", 48)),
            artifact_threshold=float(self.config.get("layout_artifact_threshold", 0.34)),
            border_band_ratio=float(self.config.get("layout_border_band_ratio", 0.055)),
            dilate_kernel=int(self.config.get("layout_dilate_kernel", 31)),
        )
        layout_mask_coverage = float((layout_mask > 0).mean()) if layout_mask.size else 0.0
        query_topology = analyze_pattern_connectivity(
            pattern_edge,
            band_ratio=float(self.config.get("topology_band_ratio", 0.12)),
            connector_width_ratio=float(self.config.get("connector_width_ratio", 0.22)),
            connector_touch_threshold=float(self.config.get("connector_touch_threshold", 0.015)),
        )
        query_closed_score = closed_shape_score(pattern_edge)

        num_orientation_bins = int(self.config.get("num_orientation_bins", 8))
        drawing_orientation = compute_gradient_orientation(drawing_gray)
        drawing_orientation_bins = quantize_orientation(drawing_orientation, num_bins=num_orientation_bins)
        distance_transforms = build_directional_distance_transforms(
            drawing_edge,
            drawing_orientation_bins,
            num_bins=num_orientation_bins,
        )

        chamfer_tau = float(self.config.get("chamfer_tau", 5.0))
        edge_tolerance = int(self.config.get("edge_tolerance", 2))
        min_chamfer_score = float(self.config.get("min_chamfer_score", 0.06))
        min_edge_f1 = float(self.config.get("min_edge_f1", 0.02))
        min_masked_precision = float(self.config.get("min_masked_precision", 0.24))
        max_outside_edge_ratio = float(self.config.get("max_outside_edge_ratio", 5.0))
        mask_dilation = int(self.config.get("mask_dilation", 2))
        outside_edge_tau = float(self.config.get("outside_edge_tau", 2.5))

        artifact_penalty_floor = float(self.config.get("artifact_penalty_floor", 0.15))
        artifact_penalty_weight = float(self.config.get("artifact_penalty_weight", 0.40))
        use_artifact_hard_gate = bool(self.config.get("use_artifact_hard_gate", False))
        min_artifact_penalty = float(self.config.get("min_artifact_penalty", 0.20))

        layout_penalty_floor = float(self.config.get("layout_penalty_floor", 0.12))
        layout_penalty_weight = float(self.config.get("layout_penalty_weight", 0.48))
        use_layout_hard_gate = bool(self.config.get("use_layout_hard_gate", False))
        min_layout_penalty = float(self.config.get("min_layout_penalty", 0.20))
        connector_width_ratio = float(self.config.get("connector_width_ratio", 0.18))
        min_query_closed_score = float(self.config.get("min_query_closed_score", 0.45))
        min_closed_compatibility = float(self.config.get("min_closed_compatibility", 0.35))

        raw_detections: List[Dict[str, Any]] = []
        candidate_scores: List[float] = []
        time_budget = float(self.config.get("time_budget_seconds", 55.0))
        verify_cutoff = max(1.0, time_budget * 0.92)
        timed_out = False

        for cand in candidates:
            if time.time() - start > verify_cutoff:
                timed_out = True
                break
            if not isinstance(cand, CandidateBox):
                continue

            x, y, w, h = int(cand.x), int(cand.y), int(cand.w), int(cand.h)
            if w <= 1 or h <= 1 or x < 0 or y < 0:
                continue
            if x + w > drawing_edge.shape[1] or y + h > drawing_edge.shape[0]:
                continue

            try:
                candidate_edge = drawing_edge[y : y + h, x : x + w]
                resized_pattern_edge = self._make_candidate_pattern_edge(pattern_edge, cand)

                pattern_orientation = compute_gradient_orientation(resized_pattern_edge.astype(np.float32))
                pattern_bins = quantize_orientation(pattern_orientation, num_bins=num_orientation_bins)
                points, point_bins = extract_oriented_edge_points(resized_pattern_edge, pattern_bins)
                if points.size == 0:
                    continue

                chamfer_distance = directional_chamfer_distance(
                    points,
                    point_bins,
                    distance_transforms,
                    x_offset=x,
                    y_offset=y,
                    soft_bins=True,
                )
                chamfer_score = chamfer_distance_to_score(chamfer_distance, tau=chamfer_tau)
                if chamfer_score < min_chamfer_score:
                    continue

                edge_iou = edge_iou_score(resized_pattern_edge, candidate_edge)
                edge_precision, edge_recall, edge_f1 = bidirectional_edge_f1(
                    resized_pattern_edge,
                    candidate_edge,
                    tolerance=edge_tolerance,
                )
                if edge_f1 < min_edge_f1:
                    continue

                masked_precision = masked_edge_precision_score(
                    resized_pattern_edge,
                    candidate_edge,
                    dilation=mask_dilation,
                )
                if masked_precision < min_masked_precision:
                    continue

                outside_edge_score, outside_edge_ratio = outside_edge_ratio_score(
                    resized_pattern_edge,
                    candidate_edge,
                    dilation=mask_dilation,
                    tau=outside_edge_tau,
                )
                connection_outside_score, connection_outside_ratio, local_topology = connection_aware_outside_edge_ratio_score(
                    resized_pattern_edge,
                    candidate_edge,
                    dilation=mask_dilation,
                    tau=outside_edge_tau,
                    connector_width_ratio=connector_width_ratio,
                    topology=query_topology,
                )
                outside_edge_score = max(outside_edge_score, connection_outside_score)
                outside_edge_ratio = min(outside_edge_ratio, connection_outside_ratio)
                if outside_edge_ratio > max_outside_edge_ratio:
                    continue

                shape_cleanliness = candidate_shape_cleanliness_score(
                    resized_pattern_edge,
                    candidate_edge,
                    dilation=mask_dilation,
                )

                candidate_edge_count = float((candidate_edge > 0).sum())
                pattern_edge_count = float((resized_pattern_edge > 0).sum())
                density = density_consistency_score(candidate_edge_count, pattern_edge_count)
                candidate_pattern_h, candidate_pattern_w = resized_pattern_edge.shape[:2]
                aspect = aspect_ratio_score(w, h, candidate_pattern_w, candidate_pattern_h)

                artifact_penalty = box_artifact_penalty(
                    artifact_mask,
                    x=x,
                    y=y,
                    w=w,
                    h=h,
                    floor=artifact_penalty_floor,
                )
                if use_artifact_hard_gate and artifact_penalty < min_artifact_penalty:
                    continue

                layout_penalty = box_layout_penalty(
                    layout_mask,
                    x=x,
                    y=y,
                    w=w,
                    h=h,
                    floor=layout_penalty_floor,
                )
                if use_layout_hard_gate and layout_penalty < min_layout_penalty:
                    continue

                candidate_closed_score = closed_shape_score(candidate_edge)
                if query_closed_score < 0.35:
                    closed_compatibility = 1.0
                else:
                    closed_compatibility = float(
                        np.exp(-abs(query_closed_score - candidate_closed_score) / 0.35)
                    )
                    closed_compatibility = float(np.clip(closed_compatibility, 0.0, 1.0))

                if query_closed_score >= min_query_closed_score and closed_compatibility < min_closed_compatibility:
                    continue

                template_score = float(getattr(cand, "template_score", 0.0))
                if template_score <= 0.0:
                    template_score = float(getattr(cand, "density_score", 0.0))

                base_score = (
                    0.24 * template_score
                    + 0.20 * chamfer_score
                    + 0.18 * edge_f1
                    + 0.17 * masked_precision
                    + 0.11 * outside_edge_score
                    + 0.05 * aspect
                    + 0.03 * edge_iou
                    + 0.02 * shape_cleanliness
                )
                confidence = 0.93 * base_score + 0.07 * density

                artifact_weight = float(np.clip(artifact_penalty_weight, 0.0, 1.0))
                confidence *= (1.0 - artifact_weight) + artifact_weight * artifact_penalty
                layout_weight = float(np.clip(layout_penalty_weight, 0.0, 1.0))
                confidence *= (1.0 - layout_weight) + layout_weight * layout_penalty
                confidence *= 0.85 + 0.15 * closed_compatibility
                confidence = float(np.clip(confidence, 0.0, 1.0))

                candidate_scores.append(confidence)
                raw_detections.append(
                    {
                        "bbox": [int(x), int(y), int(w), int(h)],
                        "confidence": confidence,
                        "scale": float(cand.scale),
                        "rotation": float(cand.rotation),
                        "scores": {
                            "template": float(template_score),
                            "chamfer": float(chamfer_score),
                            "edge_iou": float(edge_iou),
                            "edge_precision": float(edge_precision),
                            "edge_recall": float(edge_recall),
                            "edge_f1": float(edge_f1),
                            "masked_precision": float(masked_precision),
                            "outside_edge_score": float(outside_edge_score),
                            "outside_edge_ratio": float(outside_edge_ratio),
                            "connection_aware_outside_ratio": float(connection_outside_ratio),
                            "shape_cleanliness": float(shape_cleanliness),
                            "density": float(density),
                            "aspect_ratio": float(aspect),
                            "artifact_penalty": float(artifact_penalty),
                            "layout_penalty": float(layout_penalty),
                            "query_closed_score": float(query_closed_score),
                            "candidate_closed_score": float(candidate_closed_score),
                            "closed_compatibility": float(closed_compatibility),
                            "left_center_touch_ratio": float(local_topology.get("left_center_touch_ratio", 0.0)),
                            "right_center_touch_ratio": float(local_topology.get("right_center_touch_ratio", 0.0)),
                            "top_center_touch_ratio": float(local_topology.get("top_center_touch_ratio", 0.0)),
                            "bottom_center_touch_ratio": float(local_topology.get("bottom_center_touch_ratio", 0.0)),
                            "vertical_embedded": bool(local_topology.get("vertical_embedded", False)),
                            "horizontal_embedded": bool(local_topology.get("horizontal_embedded", False)),
                        },
                    }
                )
            except Exception:
                continue

        filtered, filter_meta = self._adaptive_filter(raw_detections)
        num_before_nms = len(filtered)
        detections = nms(filtered, iou_threshold=float(self.config.get("nms_iou_threshold", 0.22)))

        resized_h, resized_w = drawing_resized.shape[:2]
        original_h, original_w = drawing_image.shape[:2]
        inv_scale = 1.0 / max(scale_factor, 1e-6)

        mapped: List[Dict[str, Any]] = []
        enable_bbox_refinement = bool(self.config.get("enable_bbox_refinement", True))
        bbox_expand_ratio = float(self.config.get("bbox_expand_ratio", 0.08))
        for det in detections:
            bx, by, bw, bh = det["bbox"]
            bx = int(round(bx * inv_scale))
            by = int(round(by * inv_scale))
            bw = int(round(bw * inv_scale))
            bh = int(round(bh * inv_scale))

            bx = max(0, min(bx, original_w - 1))
            by = max(0, min(by, original_h - 1))
            bw = max(1, min(bw, original_w - bx))
            bh = max(1, min(bh, original_h - by))

            if enable_bbox_refinement and bbox_expand_ratio > 0.0:
                cx = bx + bw / 2.0
                cy = by + bh / 2.0
                new_w = max(bw, int(round(bw * (1.0 + 2.0 * bbox_expand_ratio))))
                new_h = max(bh, int(round(bh * (1.0 + 2.0 * bbox_expand_ratio))))
                bx = int(round(cx - new_w / 2.0))
                by = int(round(cy - new_h / 2.0))
                bx = max(0, min(bx, original_w - 1))
                by = max(0, min(by, original_h - 1))
                new_w = max(1, min(new_w, original_w - bx))
                new_h = max(1, min(new_h, original_h - by))
                bw, bh = new_w, new_h

            det["bbox"] = [bx, by, bw, bh]
            mapped.append(det)

        visualization = draw_detections(drawing_image, mapped)
        runtime = time.time() - start

        score_arr = np.array(candidate_scores, dtype=np.float32) if candidate_scores else np.array([], dtype=np.float32)
        metadata = {
            "runtime_seconds": float(runtime),
            "time_budget_seconds": float(time_budget),
            "verification_timed_out": bool(timed_out),
            "num_candidates": int(len(candidates)),
            "num_verified_candidates": int(len(candidate_scores)),
            "num_raw_detections": int(len(raw_detections)),
            "num_detections_before_nms": int(num_before_nms),
            "num_detections_after_nms": int(len(mapped)),
            "candidate_score_min": float(score_arr.min()) if score_arr.size else 0.0,
            "candidate_score_mean": float(score_arr.mean()) if score_arr.size else 0.0,
            "candidate_score_max": float(score_arr.max()) if score_arr.size else 0.0,
            "artifact_mask_coverage": float(artifact_mask_coverage),
            "layout_mask_coverage": float(layout_mask_coverage),
            "query_topology": query_topology,
            "query_closed_score": float(query_closed_score),
            "rotations": [float(v) for v in self.config.get("rotations", [0.0])],
            "rotations_used": [float(v) for v in self.config.get("rotations", [0.0])],
            "image_shape": [int(resized_h), int(resized_w)],
            "scale_factor": float(scale_factor),
        }
        metadata.update(filter_meta)

        return mapped, visualization, metadata
