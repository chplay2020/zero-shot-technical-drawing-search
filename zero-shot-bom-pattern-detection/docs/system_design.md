# System Design

## Pipeline Overview
1. Load and validate input images.
2. Resize to a maximum side for speed.
3. Edge detection to emphasize geometry.
4. Integral image for edge density pruning.
5. Multi-scale, small-rotation candidate generation.
6. Directional chamfer-style matching.
7. Score fusion from multiple geometric cues.
8. Non-maximum suppression.
9. Visualization and JSON output.

## Why Geometry-first
- Technical drawings are dominated by line geometry rather than texture.
- Edge maps are stable across materials, lighting, and scan quality.
- A geometry-first pipeline avoids heavy models, weights, and GPUs.
- Deterministic CPU performance fits the assessment constraints.

This design is simple to deploy, reproducible, and interpretable.
