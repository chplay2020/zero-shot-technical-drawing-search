from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DetectorConfig:
    """Configuration for the CPU-friendly zero-shot pattern detector."""

    # Search space
    scales: list[float] = field(default_factory=lambda: [0.6, 0.75, 0.9, 1.0, 1.1, 1.25, 1.4])
    rotations: list[float] = field(default_factory=lambda: [0.0, 90.0, 180.0, 270.0, -5.0, 5.0])

    # Runtime / resizing
    max_image_side: int = 1800
    max_candidates: int = 2000
    candidate_pool_size: int = 3200
    max_verification_candidates: int = 2600
    time_budget_seconds: float = 55.0
    mode: str = "accurate"

    # Template matching
    template_match_threshold: float = 0.12
    template_top_k_per_variant: int = 80
    background_penalty_tau: float = 0.18
    background_penalty_weight: float = 0.25
    enable_candidate_ensemble: bool = True
    plain_template_threshold: float = 0.08
    plain_template_top_k_per_variant: int = 120
    plain_template_max_candidates: int = 1200
    density_fallback_max_candidates: int = 600
    candidate_merge_iou: float = 0.82

    # Legacy density pruning fields kept for compatibility
    density_ratio_min: float = 0.25
    density_ratio_max: float = 4.0
    min_window_size: int = 4

    # Directional chamfer
    num_orientation_bins: int = 8
    chamfer_tau: float = 5.0

    # Verification thresholds
    min_chamfer_score: float = 0.06
    min_edge_iou: float = 0.0
    min_edge_f1: float = 0.02
    mask_dilation: int = 2
    min_masked_precision: float = 0.24
    max_outside_edge_ratio: float = 5.0
    outside_edge_tau: float = 2.5
    shape_cleanliness_weight: float = 0.25

    # Artifact / table / text suppression
    artifact_block_size: int = 64
    artifact_stride: int = 32
    artifact_threshold: float = 0.35
    artifact_dilate_kernel: int = 25
    artifact_penalty_floor: float = 0.15
    artifact_penalty_weight: float = 0.40
    use_artifact_hard_gate: bool = False
    min_artifact_penalty: float = 0.20


    # Layout / border-grid / title-block suppression
    layout_block_size: int = 96
    layout_stride: int = 48
    layout_artifact_threshold: float = 0.34
    layout_border_band_ratio: float = 0.055
    layout_dilate_kernel: int = 31
    layout_penalty_floor: float = 0.12
    layout_penalty_weight: float = 0.48
    use_layout_hard_gate: bool = False
    min_layout_penalty: float = 0.20

    # Query topology / connection-aware scoring
    topology_band_ratio: float = 0.12
    connector_width_ratio: float = 0.18
    connector_touch_threshold: float = 0.015
    min_query_closed_score: float = 0.45
    min_closed_compatibility: float = 0.35

    # Bounding box refinement
    enable_bbox_refinement: bool = True
    bbox_expand_ratio: float = 0.08

    # Search region optional mask
    search_region_block_size: int = 96
    search_region_stride: int = 48
    search_region_max_artifact_score: float = 0.55
    min_search_region_coverage: float = 0.50
    search_region_penalty_weight: float = 0.50
    use_search_region_hard_gate: bool = False

    # Dynamic filtering + NMS
    dynamic_min_threshold: float = 0.46
    dynamic_threshold_ratio: float = 0.88
    pre_nms_top_k: int = 24
    min_final_candidates_before_nms: int = 6
    enable_adaptive_recall: bool = True
    adaptive_recall_ratio: float = 0.74
    adaptive_recall_min_score: float = 0.36
    enable_empty_fallback: bool = True
    fallback_min_score: float = 0.20
    fallback_top_k: int = 8
    nms_iou_threshold: float = 0.22

    # Older threshold retained for scripts/tests that still set it.
    confidence_threshold: float = 0.0
    max_edge_excess_ratio: float = 8.0

    # Pattern preprocessing
    padding_ratio: float = 0.08

    # Optional score weights kept for compatibility
    score_weights: dict[str, float] | None = None

    @classmethod
    def from_yaml(cls, path: str | Path) -> "DetectorConfig":
        """Load detector configuration from YAML and ignore unknown keys."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        if not isinstance(data, dict):
            raise ValueError(f"Config file must contain a YAML mapping: {path}")
        valid_keys = set(cls.__dataclass_fields__.keys())
        filtered = {key: value for key, value in data.items() if key in valid_keys}
        return cls(**filtered)

    def to_dict(self) -> dict[str, Any]:
        """Convert config object to a plain dictionary."""
        return {key: getattr(self, key) for key in self.__dataclass_fields__}
