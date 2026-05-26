from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pattern_detector.config import DetectorConfig
from pattern_detector.detector import PatternDetector
from pattern_detector.preprocessing import read_image


def main() -> None:
    parser = argparse.ArgumentParser(description="Run zero-shot pattern detection.")
    parser.add_argument("--pattern", required=True, help="Path to pattern image")
    parser.add_argument("--drawing", required=True, help="Path to drawing image")
    parser.add_argument("--output", required=True, help="Path to output visualization image")
    parser.add_argument("--json-output", required=True, help="Path to output JSON file")
    parser.add_argument(
        "--config",
        default=str(ROOT / "configs" / "default.yaml"),
        help="Path to config YAML",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=None,
        help="Override confidence threshold",
    )
    parser.add_argument(
        "--advanced-search",
        action="store_true",
        help="Enable advanced search (may increase runtime)",
    )
    args = parser.parse_args()

    pattern_path = Path(args.pattern)
    drawing_path = Path(args.drawing)
    config_path = Path(args.config)
    if not pattern_path.exists():
        raise FileNotFoundError(f"Pattern image not found: {pattern_path}")
    if not drawing_path.exists():
        raise FileNotFoundError(f"Drawing image not found: {drawing_path}")
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    config = DetectorConfig.from_yaml(config_path)
    config_dict = config.to_dict()
    if args.confidence_threshold is not None:
        config_dict["confidence_threshold"] = float(args.confidence_threshold)
    config_dict["advanced_search"] = bool(args.advanced_search)

    detector = PatternDetector(config_dict)

    pattern = read_image(str(pattern_path))
    drawing = read_image(str(drawing_path))

    start = time.time()
    detections, vis_bgr, metadata = detector.detect(pattern, drawing)
    runtime = time.time() - start

    out_img = Path(args.output)
    out_json = Path(args.json_output)
    out_img.parent.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    result = {
        "detections": detections,
        "metadata": metadata,
        "image_width": int(drawing.shape[1]),
        "image_height": int(drawing.shape[0]),
    }
    out_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    cv2.imwrite(str(out_img), vis_bgr)

    print(f"Runtime: {runtime:.3f}s")
    print(f"Detections: {len(detections)}")
    print(f"Saved image: {out_img}")
    print(f"Saved JSON: {out_json}")


if __name__ == "__main__":
    main()
