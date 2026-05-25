import numpy as np

from pattern_detector.preprocessing import detect_edges


def test_detect_edges_non_empty():
    img = np.zeros((32, 32), dtype=np.float32)
    img[8:24, 16] = 1.0
    edges = detect_edges(img)
    assert edges.sum() > 0
