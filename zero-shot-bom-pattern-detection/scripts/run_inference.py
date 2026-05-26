from __future__ import annotations

import argparse
from pathlib import Path

import json

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
    parser.add_argument("--output-dir", required=True, help="Output directory")
    args = parser.parse_args()

    config = DetectorConfig.from_yaml(ROOT / "configs" / "default.yaml")
    detector = PatternDetector(config.to_dict())

    pattern = read_image(args.pattern)
    drawing = read_image(args.drawing)

    detections, vis_bgr, metadata = detector.detect(pattern, drawing)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "result.json"
    out_img = out_dir / "visualization.png"

    result = {
        "detections": detections,
        "metadata": metadata,
        "image_width": int(drawing.shape[1]),
        "image_height": int(drawing.shape[0]),
    }
    out_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    cv2.imwrite(str(out_img), vis_bgr)

    print(f"Saved: {out_json}")
    print(f"Saved: {out_img}")


if __name__ == "__main__":
    main()
