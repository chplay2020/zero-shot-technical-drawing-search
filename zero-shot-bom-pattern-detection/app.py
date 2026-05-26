from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import gradio as gr
import numpy as np
from PIL import Image

import cv2

import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from pattern_detector.config import DetectorConfig
from pattern_detector.detector import PatternDetector


def _pil_to_bgr(image: Image.Image) -> np.ndarray:
    rgb = image.convert("RGB")
    arr = np.array(rgb)
    return arr[:, :, ::-1].copy()


def _run_inference(
    pattern: Image.Image | None,
    drawing: Image.Image | None,
) -> tuple[Image.Image, Dict[str, Any]]:
    if pattern is None or drawing is None:
        raise gr.Error("Both pattern and drawing images are required.")

    config = DetectorConfig.from_yaml(ROOT / "configs" / "default.yaml")
    detector = PatternDetector(config.to_dict())

    detections, vis_bgr, metadata = detector.detect(_pil_to_bgr(pattern), _pil_to_bgr(drawing))
    vis_rgb = cv2.cvtColor(vis_bgr, cv2.COLOR_BGR2RGB)
    vis = Image.fromarray(vis_rgb)
    result = {
        "detections": detections,
        "metadata": metadata,
        "image_width": int(drawing.width),
        "image_height": int(drawing.height),
    }
    return vis, result


def main() -> None:
    demo = gr.Interface(
        fn=_run_inference,
        inputs=[
            gr.Image(type="pil", label="Pattern Image"),
            gr.Image(type="pil", label="Drawing Image"),
        ],
        outputs=[
            gr.Image(type="pil", label="Detections"),
            gr.JSON(label="Result JSON"),
        ],
        title="Zero-shot BOM Pattern Detection",
        description="Geometry-first pattern matching for technical drawings.",
    )
    demo.launch()


if __name__ == "__main__":
    main()
