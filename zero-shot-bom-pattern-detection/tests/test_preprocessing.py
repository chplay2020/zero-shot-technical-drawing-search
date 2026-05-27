import numpy as np

from pattern_detector.preprocessing import (
    binarize_image,
    detect_edges,
    to_grayscale,
    trim_pattern,
)


def test_detect_edges_non_empty():
    img = np.zeros((32, 32), dtype=np.float32)
    img[8:24, 16] = 1.0
    edges = detect_edges(img)
    assert edges.sum() > 0


def test_to_grayscale_shape():
    rgb = np.zeros((16, 20, 3), dtype=np.uint8)
    rgb[4:12, 6:14, :] = 120
    gray = to_grayscale(rgb)
    assert gray.shape == (16, 20)


def test_binarize_output_shape():
    gray = np.zeros((18, 22), dtype=np.uint8)
    gray[5:13, 8:15] = 200
    binary = binarize_image(gray, method="otsu")
    assert binary.shape == gray.shape


def test_trim_pattern_succeeds_on_synthetic():
    gray = np.zeros((30, 40), dtype=np.uint8)
    gray[8:22, 10:30] = 220
    cropped, bbox = trim_pattern(gray, padding_ratio=0.08)
    assert cropped.size > 0
    assert len(bbox) == 4
