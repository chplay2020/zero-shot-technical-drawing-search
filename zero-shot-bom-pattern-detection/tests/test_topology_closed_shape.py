import numpy as np

from pattern_detector.scoring import (
    analyze_pattern_connectivity,
    closed_shape_compatibility_score,
)


def test_connectivity_horizontal_embedded_only():
    edge = np.zeros((24, 48), dtype=np.uint8)
    edge[12, :] = 255
    topology = analyze_pattern_connectivity(
        edge,
        band_ratio=0.12,
        connector_width_ratio=0.22,
        connector_touch_threshold=0.015,
    )
    assert topology["horizontal_embedded"] is True
    assert topology["vertical_embedded"] is False


def test_connectivity_vertical_embedded_only():
    edge = np.zeros((48, 24), dtype=np.uint8)
    edge[:, 12] = 255
    topology = analyze_pattern_connectivity(
        edge,
        band_ratio=0.12,
        connector_width_ratio=0.22,
        connector_touch_threshold=0.015,
    )
    assert topology["vertical_embedded"] is True
    assert topology["horizontal_embedded"] is False


def test_connectivity_closed_shape_not_both_embedded():
    edge = np.zeros((32, 32), dtype=np.uint8)
    edge[6:26, 6] = 255
    edge[6:26, 25] = 255
    edge[6, 6:26] = 255
    edge[25, 6:26] = 255
    topology = analyze_pattern_connectivity(
        edge,
        band_ratio=0.12,
        connector_width_ratio=0.22,
        connector_touch_threshold=0.015,
    )
    assert not (topology["horizontal_embedded"] and topology["vertical_embedded"])


def test_closed_shape_compatibility_high_for_similar():
    query = np.zeros((24, 24), dtype=np.uint8)
    query[4:20, 4] = 255
    query[4:20, 19] = 255
    query[4, 4:20] = 255
    query[19, 4:20] = 255

    candidate = query.copy()
    score = closed_shape_compatibility_score(query, candidate)
    assert score > 0.8


def test_closed_shape_compatibility_penalizes_open():
    query = np.zeros((24, 24), dtype=np.uint8)
    query[4:20, 4] = 255
    query[4:20, 19] = 255
    query[4, 4:20] = 255
    query[19, 4:20] = 255

    candidate = np.zeros((24, 24), dtype=np.uint8)
    candidate[12, :] = 255

    score = closed_shape_compatibility_score(query, candidate)
    assert score < 0.7
