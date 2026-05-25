import numpy as np
import cv2
from pattern_detector.preprocessing import preprocess_pattern, preprocess_drawing

img = np.ones((200, 300), dtype=np.uint8) * 255
cv2.rectangle(img, (80, 70), (160, 120), 0, 2)

config = {
    "max_image_side": 1000,
    "padding_ratio": 0.08,
}

pattern = preprocess_pattern(img, config)
drawing = preprocess_drawing(img, config)

print("Pattern keys:", pattern.keys())
print("Drawing keys:", drawing.keys())
print("Pattern edge count:", pattern["edge_count"])
print("Drawing edge count:", drawing["edge_count"])
print("Pattern cropped shape:", pattern["cropped_edge"].shape)

assert pattern["edge_count"] > 0
assert drawing["edge_count"] > 0
assert pattern["cropped_edge"].shape[0] < img.shape[0]
assert pattern["cropped_edge"].shape[1] < img.shape[1]

print("OK")
