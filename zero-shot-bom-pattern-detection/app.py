from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

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


def _build_table(detections: List[Dict[str, Any]]) -> List[List[float]]:
    rows: List[List[float]] = []
    for det in detections:
        bbox = det.get("bbox", [0, 0, 0, 0])
        scores = det.get("scores", {})
        rows.append(
            [
                float(bbox[0]),
                float(bbox[1]),
                float(bbox[2]),
                float(bbox[3]),
                float(det.get("confidence", 0.0)),
                float(det.get("scale", 1.0)),
                float(det.get("rotation", 0.0)),
            ]
        )
    return rows


def _run_inference(
    pattern: Image.Image | None,
    drawing: Image.Image | None,
    confidence_threshold: float,
    advanced_search: bool,
    rotation_search: bool,
) -> Tuple[Image.Image | None, List[List[float]], Dict[str, Any], str]:
    if pattern is None or drawing is None:
        raise gr.Error("Please upload both the pattern image and drawing image.")

    config = DetectorConfig.from_yaml(ROOT / "configs" / "default.yaml").to_dict()
    config["confidence_threshold"] = float(confidence_threshold)
    config["advanced_search"] = bool(advanced_search)
    if not rotation_search:
        config["rotations"] = [0.0]

    detector = PatternDetector(config)

    try:
        detections, vis_bgr, metadata = detector.detect(_pil_to_bgr(pattern), _pil_to_bgr(drawing))
    except Exception as exc:
        raise gr.Error(f"Inference failed: {exc}") from exc

    vis_rgb = cv2.cvtColor(vis_bgr, cv2.COLOR_BGR2RGB)
    vis = Image.fromarray(vis_rgb)
    result = {
        "detections": detections,
        "metadata": metadata,
        "image_width": int(drawing.width),
        "image_height": int(drawing.height),
    }
    table = _build_table(detections)
    runtime = float(metadata.get("runtime_seconds", 0.0))
    meta_text = (
        f"Runtime: {runtime:.3f}s | "
        f"Detections: {len(detections)} | "
        f"Candidates: {metadata.get('num_candidates', 0)}"
    )
    return vis, table, result, meta_text


def main() -> None:
    with gr.Blocks(title="Zero-shot BOM Pattern Detection") as demo:
        gr.Markdown(
            "# Zero-shot BOM Pattern Detection\n"
            "Upload a pattern image and a drawing image to detect matching symbols."
        )

        with gr.Row():
            pattern_input = gr.Image(type="pil", label="Pattern Image")
            drawing_input = gr.Image(type="pil", label="Drawing Image")

        with gr.Row():
            confidence_slider = gr.Slider(
                minimum=0.0,
                maximum=1.0,
                step=0.01,
                value=0.5,
                label="Confidence Threshold",
            )
            advanced_checkbox = gr.Checkbox(label="Enable Advanced Search", value=False)
            rotation_checkbox = gr.Checkbox(label="Enable Rotation Search", value=True)

        run_button = gr.Button("Run Detection")

        output_image = gr.Image(type="pil", label="Detections")
        output_table = gr.Dataframe(
            headers=["x", "y", "w", "h", "confidence", "scale", "rotation"],
            datatype=["number"] * 7,
            label="Detections Table",
        )
        output_json = gr.JSON(label="Result JSON")
        output_text = gr.Textbox(label="Runtime / Metadata", interactive=False)

        run_button.click(
            fn=_run_inference,
            inputs=[
                pattern_input,
                drawing_input,
                confidence_slider,
                advanced_checkbox,
                rotation_checkbox,
            ],
            outputs=[output_image, output_table, output_json, output_text],
        )

    demo.launch()


if __name__ == "__main__":
    main()
