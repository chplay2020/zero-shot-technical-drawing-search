from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DetectorConfig:
    """Configuration for the geometry-first zero-shot pattern detector."""

    # Search space
    scales: list[float] = field(default_factory=lambda: [0.75, 0.9, 1.0, 1.1, 1.25])
    rotations: list[float] = field(default_factory=lambda: [0.0])

    # Runtime / resizing
    max_image_side: int = 1800
    max_candidates: int = 800

    # Density pruning
    density_ratio_min: float = 0.4
    density_ratio_max: float = 2.5

    # Directional chamfer
    num_orientation_bins: int = 8
    chamfer_tau: float = 4.0

    # Detection thresholds
    confidence_threshold: float = 0.55
    nms_iou_threshold: float = 0.3

    # Precision improvement filters
    min_chamfer_score: float = 0.35
    min_edge_iou: float = 0.03
    max_edge_excess_ratio: float = 3.5

    # Pattern preprocessing
    padding_ratio: float = 0.08

    # Optional score weights
    score_weights: dict[str, float] | None = None

    @classmethod
    def from_yaml(cls, path: str | Path) -> "DetectorConfig":
        """Load detector configuration from a YAML file.

        Unknown keys are ignored to make the config robust when YAML contains
        experimental parameters.
        """
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        if not isinstance(data, dict):
            raise ValueError(f"Config file must contain a YAML mapping: {path}")

        valid_keys = set(cls.__dataclass_fields__.keys())
        filtered = {k: v for k, v in data.items() if k in valid_keys}

        return cls(**filtered)

    def to_dict(self) -> dict[str, Any]:
        """Convert config object to plain dictionary."""
        return {
            "scales": self.scales,
            "rotations": self.rotations,
            "max_image_side": self.max_image_side,
            "max_candidates": self.max_candidates,
            "density_ratio_min": self.density_ratio_min,
            "density_ratio_max": self.density_ratio_max,
            "num_orientation_bins": self.num_orientation_bins,
            "chamfer_tau": self.chamfer_tau,
            "confidence_threshold": self.confidence_threshold,
            "nms_iou_threshold": self.nms_iou_threshold,
            "min_chamfer_score": self.min_chamfer_score,
            "min_edge_iou": self.min_edge_iou,
            "max_edge_excess_ratio": self.max_edge_excess_ratio,
            "padding_ratio": self.padding_ratio,
            "score_weights": self.score_weights,
        }