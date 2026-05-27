import numpy as np

from pattern_detector.integral_pruning import (
    build_integral_image,
    is_density_valid,
    window_sum,
    prune_by_density,
)


def test_window_sum_matches_expected():
    mat = np.zeros((4, 4), dtype=np.uint8)
    mat[1:3, 1:3] = 1
    integral = build_integral_image(mat)
    assert window_sum(integral, 1, 1, 2, 2) == 4.0
    assert window_sum(integral, 0, 0, 4, 4) == 4.0


def test_density_validation_bounds():
    assert is_density_valid(10.0, 10.0, min_ratio=0.5, max_ratio=2.0)
    assert not is_density_valid(1.0, 10.0, min_ratio=0.5, max_ratio=2.0)


def test_prune_by_density_keeps_expected():
    edge_map = np.zeros((10, 10), dtype=np.uint8)
    edge_map[2:5, 2:5] = 1

    integral = build_integral_image(edge_map)

    candidates = [(2, 2, 3, 3, 1.0, 0.0)]

    kept = prune_by_density(
        candidates,
        integral,
        query_edge_density=1.0,
        min_ratio=0.5,
        max_ratio=5.0,
    )

    assert len(kept) == 1