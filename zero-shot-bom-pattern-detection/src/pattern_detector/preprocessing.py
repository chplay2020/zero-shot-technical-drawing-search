from __future__ import annotations

from typing import Any, Dict, Tuple

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
    return resize_with_scale(img, max_side)


def to_gray(img_bgr: np.ndarray) -> np.ndarray:
    """Convert BGR image to float32 grayscale in [0, 1]."""
    gray = to_grayscale(img_bgr)
    return gray.astype(np.float32) / 255.0


def detect_edges(gray: np.ndarray) -> np.ndarray:
    """Compute Canny edges from a grayscale image."""
    if gray.size == 0:
        raise ValueError("Empty image provided to detect_edges")
    if gray.dtype != np.uint8:
        img_u8 = (np.clip(gray, 0.0, 1.0) * 255.0).astype(np.uint8)
    else:
        img_u8 = gray
    edges = extract_edges(img_u8, method="canny")
    return (edges > 0).astype(np.uint8)


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert input image to uint8 grayscale."""
    if image is None or image.size == 0:
        raise ValueError("Empty image provided to to_grayscale")
    if image.ndim == 2:
        gray = image
    elif image.shape[2] == 4:
        gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if gray.dtype != np.uint8:
        gray = np.clip(gray, 0, 255).astype(np.uint8)
    return gray


def normalize_polarity(gray: np.ndarray) -> np.ndarray:
    """Normalize polarity so background is white and foreground is dark."""
    if gray.ndim != 2:
        raise ValueError("normalize_polarity expects a 2D grayscale image")
    h, w = gray.shape
    border = np.concatenate(
        [gray[0, :], gray[-1, :], gray[:, 0], gray[:, -1]], axis=0
    )
    border_mean = float(border.mean()) if border.size else float(gray.mean())
    if border_mean < 128.0:
        return cv2.bitwise_not(gray)
    return gray


def binarize_image(gray: np.ndarray, method: str = "otsu") -> np.ndarray:
    """Binarize grayscale image into foreground=255 and background=0."""
    if gray.ndim != 2:
        raise ValueError("binarize_image expects a 2D grayscale image")
    if gray.dtype != np.uint8:
        gray = np.clip(gray, 0, 255).astype(np.uint8)
    if method == "otsu":
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    elif method == "adaptive":
        binary = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY_INV,
            35,
            10,
        )
    else:
        raise ValueError(f"Unsupported binarization method: {method}")
    return binary.astype(np.uint8)


def extract_edges(binary: np.ndarray, method: str = "canny") -> np.ndarray:
    """Extract binary edges from a binary image."""
    if binary.ndim != 2:
        raise ValueError("extract_edges expects a 2D binary image")
    if method == "canny":
        edges = cv2.Canny(binary, 50, 150)
    elif method == "sobel":
        gx = cv2.Sobel(binary, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(binary, cv2.CV_32F, 0, 1, ksize=3)
        mag = cv2.magnitude(gx, gy)
        edges = (mag > 0).astype(np.uint8) * 255
    else:
        raise ValueError(f"Unsupported edge method: {method}")
    return edges.astype(np.uint8)


def trim_pattern(
    binary_or_gray: np.ndarray,
    padding_ratio: float = 0.08,
) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """Trim whitespace around pattern and return cropped image and crop bbox."""
    if binary_or_gray is None or binary_or_gray.size == 0:
        raise ValueError("Empty image provided to trim_pattern")
    if binary_or_gray.ndim != 2:
        raise ValueError("trim_pattern expects a 2D image")
    if binary_or_gray.dtype != np.uint8:
        img_u8 = np.clip(binary_or_gray, 0, 255).astype(np.uint8)
    else:
        img_u8 = binary_or_gray

    if set(np.unique(img_u8)) <= {0, 255}:
        binary = img_u8
    else:
        binary = binarize_image(img_u8, method="otsu")

    ys, xs = np.where(binary > 0)
    if xs.size == 0 or ys.size == 0:
        raise ValueError("No foreground found in pattern after binarization")

    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())

    h, w = binary.shape
    pad_x = int(round((x2 - x1 + 1) * padding_ratio))
    pad_y = int(round((y2 - y1 + 1) * padding_ratio))

    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(w - 1, x2 + pad_x)
    y2 = min(h - 1, y2 + pad_y)

    cropped = binary[y1 : y2 + 1, x1 : x2 + 1]
    return cropped, (x1, y1, x2 - x1 + 1, y2 - y1 + 1)


def resize_with_scale(image: np.ndarray, max_side: int) -> Tuple[np.ndarray, float]:
    """Resize image to fit within max_side while preserving aspect ratio."""
    if image is None or image.size == 0:
        raise ValueError("Empty image provided to resize_with_scale")
    if max_side <= 0:
        raise ValueError("max_side must be positive")
    h, w = image.shape[:2]
    if max(h, w) <= max_side:
        return image, 1.0
    scale = max_side / float(max(h, w))
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized, scale


def preprocess_pattern(image: np.ndarray, config: Dict[str, Any]) -> Dict[str, Any]:
    """Preprocess pattern image for geometry-first matching."""
    if image is None or image.size == 0:
        raise ValueError("Empty pattern image")

    bin_method = str(config.get("binarization", "otsu"))
    edge_method = str(config.get("edge_method", "canny"))
    padding_ratio = float(config.get("padding_ratio", 0.08))

    original = ensure_color(image)
    gray = to_grayscale(original)
    gray = normalize_polarity(gray)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    binary = binarize_image(gray, method=bin_method)
    edge = extract_edges(binary, method=edge_method)

    cropped_binary, crop_bbox = trim_pattern(binary, padding_ratio=padding_ratio)
    cropped_edge = extract_edges(cropped_binary, method=edge_method)

    edge_count = int((cropped_edge > 0).sum())
    if edge_count == 0:
        raise ValueError("Pattern has no edges after preprocessing")

    ys, xs = np.where(binary > 0)
    if xs.size == 0 or ys.size == 0:
        raise ValueError("No foreground found in pattern after preprocessing")
    fg_bbox = (int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1))

    h, w = gray.shape
    return {
        "original": original,
        "gray": gray,
        "binary": binary,
        "edge": edge,
        "cropped_binary": cropped_binary,
        "cropped_edge": cropped_edge,
        "crop_bbox": crop_bbox,
        "foreground_bbox": fg_bbox,
        "width": int(w),
        "height": int(h),
        "edge_count": edge_count,
    }


def preprocess_drawing(image: np.ndarray, config: Dict[str, Any]) -> Dict[str, Any]:
    """Preprocess drawing image for geometry-first matching."""
    if image is None or image.size == 0:
        raise ValueError("Empty drawing image")

    bin_method = str(config.get("binarization", "otsu"))
    edge_method = str(config.get("edge_method", "canny"))
    max_side = int(config.get("max_image_side", 1600))

    original = ensure_color(image)
    resized, scale = resize_with_scale(original, max_side)

    gray = to_grayscale(resized)
    gray = normalize_polarity(gray)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    binary = binarize_image(gray, method=bin_method)
    edge = extract_edges(binary, method=edge_method)

    edge_count = int((edge > 0).sum())
    if edge_count == 0:
        raise ValueError("Drawing has no edges after preprocessing")

    h, w = gray.shape
    return {
        "original": original,
        "resized": resized,
        "scale_factor": float(scale),
        "gray": gray,
        "binary": binary,
        "edge": edge,
        "height": int(h),
        "width": int(w),
        "edge_count": edge_count,
    }
