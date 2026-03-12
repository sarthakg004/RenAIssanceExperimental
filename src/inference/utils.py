"""
Shared helpers for full-page OCR inference.
"""

from __future__ import annotations
import numpy as np
from PIL import Image
import cv2


def sort_top_to_bottom(boxes: list) -> list:
    """Sort detection polygons by their vertical centre coordinate."""
    return sorted(boxes, key=lambda b: np.array(b, dtype=float).mean(axis=0)[1])


def crop_polygon_gray(image_bgr: np.ndarray, poly_pts) -> Image.Image:
    """Crop a quadrilateral/polygon from a BGR image and return as grayscale PIL."""
    pts = np.array(poly_pts, dtype=np.float32)
    x1, y1 = pts.min(axis=0).astype(int)
    x2, y2 = pts.max(axis=0).astype(int)
    x1 = max(0, x1); y1 = max(0, y1)
    x2 = min(image_bgr.shape[1], x2)
    y2 = min(image_bgr.shape[0], y2)
    crop = image_bgr[y1:y2, x1:x2]
    return Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY))


def crop_polygon_rgb(image_bgr: np.ndarray, poly_pts) -> Image.Image:
    """Crop a quadrilateral/polygon from a BGR image and return as RGB PIL."""
    pts = np.array(poly_pts, dtype=np.float32)
    x1, y1 = pts.min(axis=0).astype(int)
    x2, y2 = pts.max(axis=0).astype(int)
    x1 = max(0, x1); y1 = max(0, y1)
    x2 = min(image_bgr.shape[1], x2)
    y2 = min(image_bgr.shape[0], y2)
    crop = image_bgr[y1:y2, x1:x2]
    return Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
