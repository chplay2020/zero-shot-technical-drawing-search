import numpy as np

from pattern_detector.config import DetectorConfig
from pattern_detector.detector import PatternDetector


def test_detector_smoke():
    config = DetectorConfig(
        scales=[1.0],
        rotations=[0.0],
        max_image_side=128,
        max_candidates=50,
        template_match_threshold=0.1,
        template_top_k_per_variant=10,
        background_penalty_tau=0.18,
        background_penalty_weight=0.30,
        artifact_block_size=32,
        artifact_stride=16,
        artifact_threshold=0.5,
        min_artifact_penalty=0.0,
        artifact_penalty_floor=0.35,
        artifact_penalty_weight=0.25,
        use_artifact_hard_gate=False,
        num_orientation_bins=8,
        chamfer_tau=3.0,
        min_chamfer_score=0.0,
        min_edge_f1=0.0,
        enable_empty_fallback=True,
        fallback_min_score=0.0,
        dynamic_min_threshold=0.0,
        dynamic_threshold_ratio=0.0,
        nms_iou_threshold=0.5,
        padding_ratio=0.08,
        mode="fast",
    )
    detector = PatternDetector(config.to_dict())

    pattern = np.zeros((24, 28, 3), dtype=np.uint8)
    pattern[4:20, 4] = 255
    pattern[4:20, 23] = 255
    pattern[4, 4:24] = 255
    pattern[19, 4:24] = 255
    drawing = np.zeros((64, 64, 3), dtype=np.uint8)
    drawing[20:44, 24:52] = pattern

    detections, vis, metadata = detector.detect(pattern, drawing)
    assert isinstance(detections, list)
    assert detections, "Expected at least one detection"
    assert vis.shape[0] == 64
    assert vis.shape[1] == 64
    assert metadata["image_shape"][0] > 0
