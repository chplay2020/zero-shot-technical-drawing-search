import numpy as np

from pattern_detector.config import DetectorConfig
from pattern_detector.detector import PatternDetector


def test_detector_smoke():
    config = DetectorConfig(
        scales=[1.0],
        rotations=[0.0],
        confidence_threshold=0.0,
        nms_iou_threshold=0.5,
        max_image_side=128,
        density_ratio_min=0.1,
        density_ratio_max=10.0,
        num_orientation_bins=8,
        chamfer_tau=3.0,
        max_candidates=50,
    )
    detector = PatternDetector(config)

    pattern = np.zeros((32, 32, 3), dtype=np.uint8)
    pattern[8:24, 16] = 255
    drawing = np.zeros((64, 64, 3), dtype=np.uint8)
    drawing[16:48, 32] = 255

    result = detector.detect(pattern, drawing)
    assert result.image_width == 64
    assert result.image_height == 64
