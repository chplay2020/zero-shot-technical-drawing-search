import numpy as np

from pattern_detector.chamfer import (
    build_directional_distance_transforms,
    chamfer_distance_to_score,
    compute_gradient_orientation,
    directional_chamfer_distance,
    extract_oriented_edge_points,
    quantize_orientation,
)


def test_chamfer_score_high_for_matching_edges():
    edge = np.zeros((24, 24), dtype=np.uint8)
    edge[6:18, 12] = 255
    orientation = compute_gradient_orientation(edge.astype(np.float32))
    bins = quantize_orientation(orientation, num_bins=8)
    dts = build_directional_distance_transforms(edge, bins, num_bins=8)
    points, point_bins = extract_oriented_edge_points(edge, bins)
    distance = directional_chamfer_distance(points, point_bins, dts, x_offset=0, y_offset=0)
    score = chamfer_distance_to_score(distance, tau=5.0)
    assert score > 0.6


def test_chamfer_score_in_unit_range():
    edge = np.zeros((20, 20), dtype=np.uint8)
    edge[5:15, 5:15] = 255
    orientation = compute_gradient_orientation(edge.astype(np.float32))
    bins = quantize_orientation(orientation, num_bins=8)
    dts = build_directional_distance_transforms(edge, bins, num_bins=8)
    points, point_bins = extract_oriented_edge_points(edge, bins)
    distance = directional_chamfer_distance(points, point_bins, dts, x_offset=0, y_offset=0)
    score = chamfer_distance_to_score(distance, tau=5.0)
    assert 0.0 <= score <= 1.0
