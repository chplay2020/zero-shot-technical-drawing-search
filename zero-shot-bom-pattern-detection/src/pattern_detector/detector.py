from __future__ import annotations

import time
from typing import List, Tuple

import cv2
import numpy as np

from pattern_detector.candidate_generator import generate_candidates
from pattern_detector.chamfer import chamfer_score, distance_transform, orientation_bins
from pattern_detector.config import DetectorConfig
from pattern_detector.integral_pruning import Candidate as RawCandidate
from pattern_detector.integral_pruning import integral_image, prune_by_density
from pattern_detector.nms import nms_indices
from pattern_detector.preprocessing import detect_edges, ensure_color, resize_max_side, to_gray
from pattern_detector.scoring import aspect_ratio_score, density_ratio_score, edge_iou, fuse_scores
from pattern_detector.schemas import BoundingBox, Detection, InferenceResult
from pattern_detector.visualization import draw_detections_legacy


class PatternDetector:
    """Geometry-first zero-shot pattern detector."""

    def __init__(self, config: DetectorConfig) -> None:
        self.config = config

    def detect(self, pattern_bgr: np.ndarray, drawing_bgr: np.ndarray) -> InferenceResult:
        """Run pattern detection and return results."""
        start = time.time()

        pattern_bgr = ensure_color(pattern_bgr)
        drawing_bgr = ensure_color(drawing_bgr)

        drawing_resized, scale = resize_max_side(
            drawing_bgr,
            self.config.max_image_side,
        )

        pattern_gray = to_gray(pattern_bgr)
        drawing_gray = to_gray(drawing_resized)

        pattern_edges = detect_edges(pattern_gray)
        drawing_edges = detect_edges(drawing_gray)

        if pattern_edges.sum() == 0 or drawing_edges.sum() == 0:
            raise ValueError("No edges found in pattern or drawing image.")

        query_density = float(pattern_edges.mean())
        integral = integral_image(drawing_edges)

        candidates = generate_candidates(
            drawing_edges.shape,
            pattern_edges.shape,
            self.config.scales,
            self.config.rotations,
            self.config.max_candidates,
        )

        raw_candidates: List[RawCandidate] = [
            (c.x, c.y, c.w, c.h, c.scale, c.rotation) for c in candidates
        ]

        pruned = prune_by_density(
            raw_candidates,
            integral,
            query_density,
            self.config.density_ratio_min,
            self.config.density_ratio_max,
        )

        dist_map = distance_transform(drawing_edges)
        orient_map = orientation_bins(drawing_gray, self.config.num_orientation_bins)
        query_orient = orientation_bins(pattern_gray, self.config.num_orientation_bins)

        boxes: List[Tuple[int, int, int, int]] = []
        scores: List[float] = []

        pattern_h, pattern_w = pattern_edges.shape[:2]

        for x, y, w, h, scale_c, rot_c in pruned:
            box = (x, y, w, h)

            chamfer = chamfer_score(
                pattern_edges,
                query_orient,
                dist_map,
                orient_map,
                box,
                self.config.num_orientation_bins,
            )

            target_edges = drawing_edges[y : y + h, x : x + w]
            resized_query_edges = cv2.resize(
                pattern_edges,
                (w, h),
                interpolation=cv2.INTER_NEAREST,
            )

            iou_score = edge_iou(resized_query_edges, target_edges)

            edge_count = float(target_edges.sum())
            candidate_density = edge_count / float(w * h)
            dens_score = density_ratio_score(query_density, candidate_density)

            ar_score = aspect_ratio_score(
                candidate_w=w,
                candidate_h=h,
                pattern_w=pattern_w,
                pattern_h=pattern_h,
            )

            score = fuse_scores(
                chamfer,
                iou_score,
                dens_score,
                ar_score,
            )

            if score >= self.config.confidence_threshold:
                boxes.append(box)
                scores.append(score)

        keep = nms_indices(boxes, scores, self.config.nms_iou_threshold)
        boxes = [boxes[i] for i in keep]
        scores = [scores[i] for i in keep]

        if scale != 1.0:
            inv = 1.0 / scale
            boxes = [
                (
                    int(round(x * inv)),
                    int(round(y * inv)),
                    int(round(w * inv)),
                    int(round(h * inv)),
                )
                for x, y, w, h in boxes
            ]

        vis = draw_detections_legacy(drawing_bgr, boxes, scores)
        runtime_ms = (time.time() - start) * 1000.0

        detections = [
            Detection(
                bbox=BoundingBox(x=b[0], y=b[1], w=b[2], h=b[3]),
                score=s,
            )
            for b, s in zip(boxes, scores)
        ]

        result = InferenceResult(
            detections=detections,
            image_width=int(drawing_bgr.shape[1]),
            image_height=int(drawing_bgr.shape[0]),
            runtime_ms=runtime_ms,
            visualization_rgb=vis[:, :, ::-1].tolist(),
        )

        return result