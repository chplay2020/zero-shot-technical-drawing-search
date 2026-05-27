import cv2
import numpy as np

from pattern_detector.candidate_generator import generate_template_matching_candidates

drawing_edge = np.zeros((300, 400), dtype=np.uint8)
cv2.rectangle(drawing_edge, (50, 60), (130, 110), 255, 2)
cv2.rectangle(drawing_edge, (220, 150), (300, 200), 255, 2)

pattern_edge = np.zeros((50, 80), dtype=np.uint8)
cv2.rectangle(pattern_edge, (0, 0), (79, 49), 255, 2)

config = {
    "scales": [1.0],
    "rotations": [0.0],
    "max_candidates": 20,
    "template_match_threshold": 0.2,
    "template_top_k_per_variant": 20,
}

candidates = generate_template_matching_candidates(drawing_edge, pattern_edge, config)

print("num candidates:", len(candidates))
print(candidates[:5])

assert len(candidates) > 0
assert candidates[0].template_score >= candidates[-1].template_score

print("OK")
