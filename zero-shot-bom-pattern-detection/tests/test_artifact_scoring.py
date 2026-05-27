import numpy as np

from pattern_detector.scoring import (
    artifact_penalty_score,
    box_artifact_penalty,
    build_artifact_mask,
    component_clutter_score,
    edge_density_score,
    line_grid_score,
    masked_edge_precision_score,
)


def test_artifact_scores_range():
    edge = np.zeros((32, 32), dtype=np.uint8)
    edge[8:24, 16] = 255
    assert 0.0 <= edge_density_score(edge) <= 1.0
    assert 0.0 <= line_grid_score(edge) <= 1.0
    assert 0.0 <= component_clutter_score(edge) <= 1.0
    assert 0.0 <= artifact_penalty_score(edge) <= 1.0


def test_artifact_mask_and_box_penalty():
    edge = np.zeros((64, 64), dtype=np.uint8)
    edge[10:54, 10] = 255
    edge[10:54, 20] = 255
    edge[10, 10:54] = 255
    edge[20, 10:54] = 255
    mask = build_artifact_mask(edge, block_size=32, stride=16, threshold=0.5)
    penalty = box_artifact_penalty(mask, 0, 0, 32, 32)
    assert 0.0 <= penalty <= 1.0


def test_masked_edge_precision_score_penalizes_extra_edges():
    pattern = np.zeros((20, 24), dtype=np.uint8)
    pattern[3:17, 3] = 255
    pattern[3:17, 20] = 255
    pattern[3, 3:21] = 255
    pattern[16, 3:21] = 255

    clean_candidate = pattern.copy()
    noisy_candidate = pattern.copy()
    noisy_candidate[0:5, 0:24] = 255
    noisy_candidate[18:20, 0:24] = 255
    noisy_candidate[:, 10] = 255

    clean_score = masked_edge_precision_score(pattern, clean_candidate, dilation=1)
    noisy_score = masked_edge_precision_score(pattern, noisy_candidate, dilation=1)

    assert clean_score > noisy_score
    assert clean_score > 0.8
