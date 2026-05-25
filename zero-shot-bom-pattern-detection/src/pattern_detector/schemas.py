from __future__ import annotations

from typing import Any, List

from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    """Axis-aligned bounding box."""

    x: int = Field(..., ge=0)
    y: int = Field(..., ge=0)
    w: int = Field(..., ge=1)
    h: int = Field(..., ge=1)


class Detection(BaseModel):
    """Single pattern detection result."""

    bbox: BoundingBox
    score: float = Field(..., ge=0.0, le=1.0)
    label: str = "pattern"


class InferenceResult(BaseModel):
    """Result container for inference output."""

    detections: List[Detection]
    image_width: int
    image_height: int
    runtime_ms: float
    visualization_rgb: List[List[List[int]]]

    def to_dict(self) -> dict[str, Any]:
        """Convert to a JSON-serializable dict."""
        if hasattr(self, "model_dump"):
            data = self.model_dump()
        else:
            data = dict(self.__dict__)
        data["visualization_rgb"] = None
        return data
