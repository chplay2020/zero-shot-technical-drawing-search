# System Design

## Pipeline Overview
1. Load and validate input images.
2. Resize drawing to a maximum side for speed.
3. Preprocess edges: grayscale, polarity normalization, binarization, edge extraction.
4. Template matching-first candidate generation (multi-scale, optional rotations).
5. Verification tier using directional chamfer + edge overlap metrics.
6. Dynamic threshold based on best candidate score.
7. Non-maximum suppression to remove overlaps.
8. Visualization and JSON output.

## Why Not Deep Learning
- The task uses line drawings with minimal texture, where geometry wins.
- CPU-only constraints and no model weights forbid large foundation models.
- Zero-shot requirement rules out training or fine-tuning.

## Selected Approach
Hybrid Edge Template Matching with Directional Chamfer Verification.

## Verification Tier
- Directional chamfer distance against orientation-aware DTs.
- Edge IoU and edge F1 to reject text/box clutter.
- Density and aspect ratio checks to reduce false positives.

## Notes on Deployment
- Pure CPU, deterministic runtime.
- No external weights.
- Easy to deploy on HuggingFace Spaces with Gradio.

## Limitations
- Template matching is sensitive to extreme scale/rotation not covered by configs.
- Dense text regions can still trigger false positives in rare cases.

## Future Work
- Adaptive rotation search and learned priors for symbol layout.
- Better context suppression around tables and title blocks.
