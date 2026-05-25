from pattern_detector.nms import nms


def test_nms_keeps_high_score():
    boxes = [(0, 0, 10, 10), (1, 1, 10, 10)]
    scores = [0.9, 0.5]
    keep = nms(boxes, scores, 0.5)
    assert keep == [0]
