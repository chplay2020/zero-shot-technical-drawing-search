import numpy as np
import cv2

from pattern_detector.scoring import (
    bidirectional_edge_f1,
    contour_circularity_score,
    count_small_components,
    edge_centroid_score,
    edge_coverage_score,
    extra_edge_ratio_score,
    grid_artifact_score,
    small_component_score,
)

circle = np.zeros((80, 80), dtype=np.uint8)
cv2.circle(circle, (40, 40), 20, 255, 2)

noise = np.zeros((80, 80), dtype=np.uint8)
for i in range(5, 70, 10):
    cv2.rectangle(noise, (i, 30), (i + 4, 40), 255, 1)

circle_score = contour_circularity_score(circle)
noise_score = contour_circularity_score(noise)

print("circle_score:", circle_score)
print("noise_score:", noise_score)

assert 0.0 <= circle_score <= 1.0
assert 0.0 <= noise_score <= 1.0
assert circle_score > noise_score

coverage = edge_coverage_score(circle, circle, tolerance=2)
precision, recall, f1 = bidirectional_edge_f1(circle, circle, tolerance=2)

print("coverage:", coverage)
print("precision:", precision)
print("recall:", recall)
print("f1:", f1)

assert coverage > 0.95
assert precision > 0.95
assert recall > 0.95
assert f1 > 0.95

components = count_small_components(noise, min_area=3)
component_score = small_component_score(noise, max_components=3, min_area=3)

print("components:", components)
print("component_score:", component_score)

assert components > 0
assert 0.0 <= component_score <= 1.0

centroid = edge_centroid_score(circle, circle)
extra = extra_edge_ratio_score(100, 100)
grid_score = grid_artifact_score(noise)

print("centroid:", centroid)
print("extra:", extra)
print("grid_score:", grid_score)

assert centroid > 0.95
assert extra > 0.95
assert 0.0 <= grid_score <= 1.0

print("OK")
