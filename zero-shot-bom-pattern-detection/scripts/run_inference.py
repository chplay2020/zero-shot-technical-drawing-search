from __future__ import annotations

import argparse
from pathlib import Path

import json

import cv2
import numpy as np

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
    detector = PatternDetector(config)

    pattern = read_image(args.pattern)
    drawing = read_image(args.drawing)

    result = detector.detect(pattern, drawing)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "result.json"
    out_img = out_dir / "visualization.png"

    out_json.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
    vis = cv2.cvtColor(drawing, cv2.COLOR_BGR2RGB)
    if result.visualization_rgb:
        vis = np.array(result.visualization_rgb, dtype=np.uint8)
    cv2.imwrite(str(out_img), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))

    print(f"Saved: {out_json}")
    print(f"Saved: {out_img}")


if __name__ == "__main__":
    main()
