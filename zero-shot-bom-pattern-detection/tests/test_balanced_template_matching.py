import numpy as np

from pattern_detector.candidate_generator import generate_balanced_template_candidates


def test_generate_balanced_template_candidates():
    drawing = np.zeros((64, 64), dtype=np.uint8)
    pattern = np.zeros((8, 8), dtype=np.uint8)
    pattern[2:6, 4] = 255
    pattern[4, 2:6] = 255

    y0, x0 = 24, 18
    drawing[y0 : y0 + pattern.shape[0], x0 : x0 + pattern.shape[1]] = pattern

    config = {
        "scales": [1.0],
        "rotations": [0.0],
        "max_candidates": 20,
        "template_match_threshold": 0.1,
        "template_top_k_per_variant": 10,
        "background_penalty_tau": 0.18,
        "background_penalty_weight": 0.30,
    }

    candidates = generate_balanced_template_candidates(drawing, pattern, config)
    assert candidates, "Expected at least one candidate"
    assert any(c.x == x0 and c.y == y0 for c in candidates)


def test_generate_balanced_template_candidates_rectangle():
    drawing = np.zeros((80, 80), dtype=np.uint8)
    pattern = np.zeros((12, 16), dtype=np.uint8)
    pattern[2:10, 2] = 255
    pattern[2:10, 13] = 255
    pattern[2, 2:14] = 255
    pattern[9, 2:14] = 255

    y0, x0 = 30, 28
    drawing[y0 : y0 + pattern.shape[0], x0 : x0 + pattern.shape[1]] = pattern

    config = {
        "scales": [1.0],
        "rotations": [0.0],
        "max_candidates": 30,
        "template_match_threshold": 0.1,
        "template_top_k_per_variant": 10,
        "background_penalty_tau": 0.18,
        "background_penalty_weight": 0.30,
    }

    candidates = generate_balanced_template_candidates(drawing, pattern, config)
    assert candidates, "Expected at least one candidate"
    assert any(c.x == x0 and c.y == y0 for c in candidates)
