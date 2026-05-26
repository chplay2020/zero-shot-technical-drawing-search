from __future__ import annotations

import time
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np

from pattern_detector.candidate_generator import CandidateBox, generate_candidates_by_density
from pattern_detector.chamfer import (
    build_directional_distance_transforms,
    chamfer_distance_to_score,
    compute_gradient_orientation,
    directional_chamfer_distance,
    extract_oriented_edge_points,
    quantize_orientation,
    transform_points,
)
from pattern_detector.nms import nms
from pattern_detector.preprocessing import preprocess_drawing, preprocess_pattern
from pattern_detector.scoring import (
    aspect_ratio_score,
    density_consistency_score,
    edge_iou_score,
    fuse_scores,
)
from pattern_detector.visualization import draw_detections


class PatternDetector:
    """Orchestrates the geometry-first zero-shot pattern detection pipeline."""

    def __init__(self, config: Any) -> None:
        """Initialize detector with either a dict or a simple config object."""
        if isinstance(config, dict):
            self.config = config.copy()
        elif hasattr(config, "__dict__"):
            self.config = vars(config).copy()
        else:
            raise TypeError(
                "Unsupported config type. Expected dict or config object with __dict__."
            )

    def detect(
        self,
        pattern_image: np.ndarray,
        drawing_image: np.ndarray,
    ) -> Tuple[List[Dict[str, Any]], np.ndarray, Dict[str, Any]]:
        """Run full detection pipeline and return detections, visualization, metadata."""
        start = time.time()

        if pattern_image is None or pattern_image.size == 0:
            raise ValueError("Empty pattern image provided to detect.")
        if drawing_image is None or drawing_image.size == 0:
            raise ValueError("Empty drawing image provided to detect.")

        pattern_data = preprocess_pattern(pattern_image, self.config)
        drawing_data = preprocess_drawing(drawing_image, self.config)

        pattern_edge = pattern_data["cropped_edge"]
        pattern_binary = pattern_data["cropped_binary"]

        pattern_h, pattern_w = pattern_edge.shape[:2]
        pattern_edge_count = float((pattern_edge > 0).sum())

        drawing_edge = drawing_data["edge"]
        drawing_gray = drawing_data["gray"]
        drawing_resized = drawing_data["resized"]
        scale_factor = float(drawing_data["scale_factor"])

        candidates = generate_candidates_by_density(
            drawing_edge,
            pattern_edge,
            self.config,
        )

        num_orientation_bins = int(self.config.get("num_orientation_bins", 8))

        orientation = compute_gradient_orientation(drawing_gray)
        orient_bins = quantize_orientation(
            orientation,
            num_bins=num_orientation_bins,
        )

        distance_transforms = build_directional_distance_transforms(
            drawing_edge,
            orient_bins,
            num_bins=num_orientation_bins,
        )

        pattern_orientation = compute_gradient_orientation(
            pattern_binary.astype(np.float32)
        )
        pattern_bins = quantize_orientation(
            pattern_orientation,
            num_bins=num_orientation_bins,
        )

        points, point_bins = extract_oriented_edge_points(
            pattern_edge,
            pattern_bins,
        )

        if points.size == 0:
            raise ValueError("No pattern edge points available for matching.")

        detections: List[Dict[str, Any]] = []

        chamfer_tau = float(self.config.get("chamfer_tau", 4.0))
        conf_threshold = float(self.config.get("confidence_threshold", 0.5))
        weights = self.config.get("score_weights", None)

        for cand in candidates:
            if not isinstance(cand, CandidateBox):
                continue

            x, y, w, h = cand.x, cand.y, cand.w, cand.h

            if w <= 1 or h <= 1:
                continue
            if x < 0 or y < 0:
                continue
            if x + w > drawing_edge.shape[1] or y + h > drawing_edge.shape[0]:
                continue

            center = (pattern_w / 2.0, pattern_h / 2.0)

            transformed_points = transform_points(
                points,
                center,
                cand.scale,
                cand.rotation,
            )

            chamfer_dist = directional_chamfer_distance(
                transformed_points,
                point_bins,
                distance_transforms,
                x_offset=x,
                y_offset=y,
                soft_bins=True,
            )

            chamfer_score = chamfer_distance_to_score(
                chamfer_dist,
                tau=chamfer_tau,
            )

            candidate_edge = drawing_edge[y : y + h, x : x + w]

            resized_pattern_edge = cv2.resize(
                pattern_edge,
                (w, h),
                interpolation=cv2.INTER_NEAREST,
            )

            edge_iou = edge_iou_score(
                resized_pattern_edge,
                candidate_edge,
            )

            candidate_edge_count = float((candidate_edge > 0).sum())

            density = density_consistency_score(
                candidate_edge_count,
                pattern_edge_count,
            )

            aspect = aspect_ratio_score(
                candidate_w=w,
                candidate_h=h,
                pattern_w=pattern_w,
                pattern_h=pattern_h,
            )

            confidence = fuse_scores(
                chamfer_score=chamfer_score,
                edge_iou=edge_iou,
                density=density,
                aspect=aspect,
                weights=weights,
            )

            if confidence < conf_threshold:
                continue

            detections.append(
                {
                    "bbox": [int(x), int(y), int(w), int(h)],
                    "confidence": float(confidence),
                    "scale": float(cand.scale),
                    "rotation": float(cand.rotation),
                    "scores": {
                        "chamfer": float(chamfer_score),
                        "edge_iou": float(edge_iou),
                        "density": float(density),
                        "aspect_ratio": float(aspect),
                    },
                }
            )

        num_before_nms = len(detections)

        detections = nms(
            detections,
            iou_threshold=float(self.config.get("nms_iou_threshold", 0.3)),
        )

        resized_h, resized_w = drawing_resized.shape[:2]
        orig_h, orig_w = drawing_image.shape[:2]

        inv_scale = 1.0 / max(scale_factor, 1e-6)

        mapped: List[Dict[str, Any]] = []

        for det in detections:
            x, y, w, h = det["bbox"]

            x = int(round(x * inv_scale))
            y = int(round(y * inv_scale))
            w = int(round(w * inv_scale))
            h = int(round(h * inv_scale))

            x = max(0, min(x, orig_w - 1))
            y = max(0, min(y, orig_h - 1))
            w = max(1, min(w, orig_w - x))
            h = max(1, min(h, orig_h - y))

            det["bbox"] = [x, y, w, h]
            mapped.append(det)

        vis = draw_detections(drawing_image, mapped)

        runtime = time.time() - start

        metadata = {
            "runtime_seconds": float(runtime),
            "num_candidates": int(len(candidates)),
            "num_detections_before_nms": int(num_before_nms),
            "num_detections_after_nms": int(len(mapped)),
            "image_shape": [int(resized_h), int(resized_w)],
            "scale_factor": float(scale_factor),
        }

        return mapped, vis, metadata