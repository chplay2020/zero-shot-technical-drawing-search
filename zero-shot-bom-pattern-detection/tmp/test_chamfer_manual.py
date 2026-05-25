import numpy as np
import cv2
from pattern_detector.chamfer import (
    compute_gradient_orientation,
    quantize_orientation,
    extract_oriented_edge_points,
    build_directional_distance_transforms,
    directional_chamfer_distance,
    chamfer_distance_to_score,
)

edge = np.zeros((100, 100), dtype=np.uint8)
cv2.rectangle(edge, (30, 30), (70, 60), 255, 2)

orientation = compute_gradient_orientation(edge)
bins = quantize_orientation(orientation, num_bins=8)
points, point_bins = extract_oriented_edge_points(edge, bins)
dts = build_directional_distance_transforms(edge, bins, num_bins=8)

dist = directional_chamfer_distance(
    points=points,
    point_bins=point_bins,
    distance_transforms=dts,
    x_offset=0,
    y_offset=0,
    soft_bins=True,
)

score = chamfer_distance_to_score(dist, tau=4.0)

print("num points:", len(points))
print("distance:", dist)
print("score:", score)

assert len(points) > 0
assert 0 <= score <= 1
assert score > 0.7, "Same shape should have high chamfer score"

print("OK")
