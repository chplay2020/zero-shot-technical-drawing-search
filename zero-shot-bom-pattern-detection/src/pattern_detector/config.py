from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import yaml


@dataclass(frozen=True)
class DetectorConfig:
    """Configuration for geometry-first pattern detection."""

    scales: List[float]
    rotations: List[float]
    confidence_threshold: float
    nms_iou_threshold: float
    max_image_side: int
    density_ratio_min: float
    density_ratio_max: float
    num_orientation_bins: int
    chamfer_tau: float
    max_candidates: int

    @staticmethod
    def from_yaml(path: Path) -> "DetectorConfig":
        """Load config from a YAML file."""
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return DetectorConfig(**data)

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to a JSON-serializable dictionary."""
        return {
            "scales": list(self.scales),
            "rotations": list(self.rotations),
            "confidence_threshold": float(self.confidence_threshold),
            "nms_iou_threshold": float(self.nms_iou_threshold),
            "max_image_side": int(self.max_image_side),
            "density_ratio_min": float(self.density_ratio_min),
            "density_ratio_max": float(self.density_ratio_max),
            "num_orientation_bins": int(self.num_orientation_bins),
            "chamfer_tau": float(self.chamfer_tau),
            "max_candidates": int(self.max_candidates),
        }
