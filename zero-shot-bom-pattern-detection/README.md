# Zero-shot BOM Pattern Detection

Geometry-first zero-shot pattern detection for technical drawings. The system accepts a query pattern image and a full drawing image, then returns bounding boxes where the pattern appears. No training, no fine-tuning, and CPU-only execution.

## Features
- Edge-based preprocessing
- Integral-image edge density pruning
- Multi-scale and small-rotation candidate generation
- Directional chamfer-style matching
- Score fusion (chamfer, edge IoU, density, aspect ratio)
- Non-maximum suppression
- Gradio demo for HuggingFace Spaces

## Install
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run CLI
```bash
python scripts/run_inference.py \
  --pattern examples/patterns/pattern.png \
  --drawing examples/drawings/drawing.png \
  --output-dir examples/outputs
```

## Run Gradio App
```bash
python app.py
```

## Output Format
The detector returns a JSON result with:
- `detections`: list of {bbox, score, label}
- `image_width`, `image_height`
- `runtime_ms`

Each `bbox` has {x, y, w, h} in drawing coordinates.
