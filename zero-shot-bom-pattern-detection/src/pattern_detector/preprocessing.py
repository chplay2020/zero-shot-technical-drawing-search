from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np


def read_image(path: str) -> np.ndarray:
    """Read an image from disk in BGR format."""
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None or img.size == 0:
        raise ValueError(f"Failed to read image: {path}")
    return img


def ensure_color(img: np.ndarray) -> np.ndarray:
    """Ensure image is 3-channel BGR."""
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img


def resize_max_side(img: np.ndarray, max_side: int) -> Tuple[np.ndarray, float]:
    """Resize image so the longest side equals max_side."""
    h, w = img.shape[:2]
    if max(h, w) <= max_side:
        return img, 1.0
    scale = max_side / float(max(h, w))
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized, scale


def to_gray(img_bgr: np.ndarray) -> np.ndarray:
    """Convert BGR image to float32 grayscale in [0, 1]."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return gray.astype(np.float32) / 255.0


def detect_edges(gray: np.ndarray) -> np.ndarray:
    """Compute Canny edges from a grayscale image."""
    if gray.size == 0:
        raise ValueError("Empty image provided to detect_edges")
    img_u8 = (np.clip(gray, 0.0, 1.0) * 255.0).astype(np.uint8)
    median = float(np.median(img_u8))
    lower = int(max(0, 0.66 * median))
    upper = int(min(255, 1.33 * median))
    edges = cv2.Canny(img_u8, lower, upper)
    return (edges > 0).astype(np.uint8)
