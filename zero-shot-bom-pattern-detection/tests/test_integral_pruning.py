import numpy as np

from pattern_detector.integral_pruning import integral_image, prune_by_density


def test_prune_by_density_keeps_expected():
    edge_map = np.zeros((10, 10), dtype=np.uint8)
    edge_map[2:5, 2:5] = 1

    integral = integral_image(edge_map)

    # Candidate window covers exactly the 3x3 foreground block.
    # Therefore its edge density is 9 / 9 = 1.0.
    candidates = [(2, 2, 3, 3, 1.0, 0.0)]

    kept = prune_by_density(
        candidates,
        integral,
        query_edge_density=1.0,
        min_ratio=0.5,
        max_ratio=5.0,
    )

    assert len(kept) == 1