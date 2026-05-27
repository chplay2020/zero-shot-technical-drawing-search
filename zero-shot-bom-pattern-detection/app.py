from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

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


def _apply_mode_overrides(config: Dict[str, Any], mode: str) -> Dict[str, Any]:
    if mode == "Accurate":
        return {
            **config,
            "scales": [0.6, 0.75, 0.9, 1.0, 1.1, 1.25, 1.4],
            "rotations": [0.0, -5.0, 5.0],
            "max_candidates": 3000,
            "template_top_k_per_variant": 400,
            "max_image_side": 2200,
            "mode": "accurate",
        }
    return {
        **config,
        "scales": [0.8, 0.9, 1.0, 1.1, 1.2],
        "rotations": [0.0],
        "max_candidates": 500,
        "template_top_k_per_variant": 80,
        "max_image_side": 1600,
        "mode": "fast",
    }


def _run_inference(
    pattern: Image.Image | None,
    drawing: Image.Image | None,
    dynamic_threshold_ratio: float,
    dynamic_min_threshold: float,
    mode: str,
) -> Tuple[Image.Image, Dict[str, Any], Dict[str, Any]]:
    if pattern is None or drawing is None:
        raise gr.Error("Both pattern and drawing images are required.")

    config = DetectorConfig.from_yaml(ROOT / "configs" / "default.yaml").to_dict()
    config = _apply_mode_overrides(config, mode)
    config["dynamic_threshold_ratio"] = float(dynamic_threshold_ratio)
    config["dynamic_min_threshold"] = float(dynamic_min_threshold)

    detector = PatternDetector(config)

    detections, vis_bgr, metadata = detector.detect(_pil_to_bgr(pattern), _pil_to_bgr(drawing))
    vis_rgb = cv2.cvtColor(vis_bgr, cv2.COLOR_BGR2RGB)
    vis = Image.fromarray(vis_rgb)
    result = {
        "detections": detections,
        "metadata": metadata,
        "image_width": int(drawing.width),
        "image_height": int(drawing.height),
    }
    return vis, result, metadata


def main() -> None:
    demo = gr.Interface(
        fn=_run_inference,
        inputs=[
            gr.Image(type="pil", label="Pattern Image"),
            gr.Image(type="pil", label="Drawing Image"),
            gr.Slider(
                minimum=0.5,
                maximum=0.9,
                step=0.01,
                value=0.72,
                label="Dynamic Threshold Ratio",
            ),
            gr.Slider(
                minimum=0.1,
                maximum=0.6,
                step=0.01,
                value=0.32,
                label="Dynamic Min Threshold",
            ),
            gr.Dropdown(
                choices=["Fast", "Accurate"],
                value="Accurate",
                label="Search Mode",
            ),
        ],
        outputs=[
            gr.Image(type="pil", label="Detections"),
            gr.JSON(label="Result JSON"),
            gr.JSON(label="Metadata"),
        ],
        title="Zero-shot BOM Pattern Detection",
        description="Template matching-first hybrid pipeline for technical drawings.",
    )
    demo.launch()


if __name__ == "__main__":
    main()
