# Zero-shot BOM Pattern Detection

Template matching-first hybrid pipeline for zero-shot pattern detection in technical BOM drawings. The system accepts a query pattern image and a full drawing image, then returns bounding boxes where the pattern appears. It runs on CPU, uses no training or model weights, and is suitable for HuggingFace Spaces deployment.

## Problem Statement
Given a query pattern image and a large technical drawing, find all pattern occurrences and return bounding boxes, confidence scores, visualization image, and JSON output. The solution must be zero-shot, CPU-friendly, and deterministic.

## Why Not Deep Learning
- Technical drawings are line-dominated and have low texture; classical geometry is strong here.
- Heavy foundation models require GPUs and weights, which violate CPU-only constraints.
- Training or fine-tuning is disallowed in this assessment.
- A deterministic pipeline is easier to deploy, debug, and evaluate.

## Selected Approach
Hybrid Edge Template Matching with Directional Chamfer Verification.

## Pipeline
- Preprocessing: grayscale, normalize polarity, binarize, edge extraction, trim pattern.
- Multi-scale edge template matching: generate top candidate boxes efficiently.
- Candidate verification: directional chamfer + edge IoU + edge F1 + density/aspect checks.
- Dynamic thresholding: adapt confidence cutoff to the best candidate.
- NMS: remove overlapping detections.
- Gradio demo for interactive use.

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
  --output examples/outputs/result.png \
  --json-output examples/outputs/result.json
```

## Run Gradio App
```bash
python app.py
```

## Output JSON Format
The detector returns a JSON result with:
- `detections`: list of detections, each with:
  - `bbox`: `[x, y, w, h]`
  - `confidence`
  - `scale`, `rotation`
  - `scores`: per-candidate metrics (template, chamfer, edge_iou, edge_f1, density, aspect_ratio)
- `metadata`: runtime and pipeline statistics
- `image_width`, `image_height`

## Limitations
- Template matching can be sensitive to scan artifacts and extreme rotations.
- Very dense text regions can still produce false positives on some drawings.
- Highly stylized symbols may require tuned scales/thresholds.

## Future Work
- Add rotation-invariant template matching or coarse rotation sweeps.
- Smarter context suppression for dense tables and title blocks.
- Pattern clustering for repeated symbols across sheets.
