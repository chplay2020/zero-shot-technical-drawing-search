import numpy as np
import cv2
from pattern_detector.config import DetectorConfig
from pattern_detector.detector import PatternDetector

pattern = np.ones((80, 100), dtype=np.uint8) * 255
cv2.rectangle(pattern, (20, 25), (80, 55), 0, 2)

drawing = np.ones((400, 600), dtype=np.uint8) * 255
cv2.rectangle(drawing, (120, 100), (180, 130), 0, 2)
cv2.rectangle(drawing, (350, 250), (410, 280), 0, 2)

config = DetectorConfig(
    max_image_side=1200,
    scales=[0.8, 1.0, 1.2],
    rotations=[0.0],
    confidence_threshold=0.5,
    nms_iou_threshold=0.3,
    density_ratio_min=0.1,
    density_ratio_max=10.0,
    num_orientation_bins=8,
    chamfer_tau=4.0,
    max_candidates=300,
)

detector = PatternDetector(config)
detections, vis, metadata = detector.detect(pattern, drawing)

print("metadata:", metadata)
print("num detections:", len(detections))
print("detections:", detections)
cv2.imwrite("tmp/detector_result_threshold_05.png", vis)

assert len(detections) >= 1
print("OK")
