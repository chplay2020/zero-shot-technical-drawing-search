import numpy as np

from pattern_detector.chamfer import chamfer_score, distance_transform, orientation_bins


def test_chamfer_score_basic():
    edges = np.zeros((16, 16), dtype=np.uint8)
    edges[4:12, 8] = 1
    dist = distance_transform(edges)
    orient = orientation_bins(edges.astype(np.float32), 8)
    score = chamfer_score(edges, orient, dist, orient, (0, 0, 16, 16), 8)
    assert score >= 0.0
