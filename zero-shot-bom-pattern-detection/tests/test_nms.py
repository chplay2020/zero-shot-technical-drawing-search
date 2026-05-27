from pattern_detector.nms import nms, nms_indices


def test_nms_keeps_high_score():
    boxes = [(0, 0, 10, 10), (1, 1, 10, 10)]
    scores = [0.9, 0.5]
    keep = nms_indices(boxes, scores, 0.5)
    assert keep == [0]


def test_nms_removes_overlap_detection_dicts():
    detections = [
        {"bbox": [0, 0, 10, 10], "confidence": 0.9},
        {"bbox": [1, 1, 10, 10], "confidence": 0.4},
        {"bbox": [30, 30, 8, 8], "confidence": 0.7},
    ]
    kept = nms(detections, iou_threshold=0.5)
    assert len(kept) == 2
    assert kept[0]["confidence"] >= kept[1]["confidence"]
