from __future__ import annotations

import argparse
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
    parser = argparse.ArgumentParser(description="Benchmark zero-shot detector.")
    parser.add_argument("--pattern", required=True, help="Path to pattern image")
    parser.add_argument("--drawing", required=True, help="Path to drawing image")
    parser.add_argument("--runs", type=int, default=3, help="Number of runs")
    args = parser.parse_args()

    config = DetectorConfig.from_yaml(ROOT / "configs" / "default.yaml")
    detector = PatternDetector(config.to_dict())

    pattern = read_image(args.pattern)
    drawing = read_image(args.drawing)

    times = []
    for _ in range(args.runs):
        start = time.time()
        detector.detect(pattern, drawing)
        times.append((time.time() - start) * 1000.0)

    avg_ms = sum(times) / len(times)
    print(f"Average runtime: {avg_ms:.2f} ms over {args.runs} runs")


if __name__ == "__main__":
    main()
