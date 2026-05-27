import numpy as np

from pattern_detector.candidate_generator import generate_template_matching_candidates


def test_generate_template_matching_candidates_finds_match():
    drawing = np.zeros((64, 64), dtype=np.uint8)
    pattern = np.zeros((8, 8), dtype=np.uint8)
    pattern[2:6, 4] = 255
    pattern[4, 2:6] = 255

    y0, x0 = 30, 20
    drawing[y0 : y0 + pattern.shape[0], x0 : x0 + pattern.shape[1]] = pattern

    config = {
        "scales": [1.0],
        "rotations": [0.0],
        "max_candidates": 50,
        "template_match_threshold": 0.2,
        "template_top_k_per_variant": 10,
    }

    candidates = generate_template_matching_candidates(drawing, pattern, config)
    assert candidates, "Expected at least one candidate"
    assert any(c.x == x0 and c.y == y0 for c in candidates)
