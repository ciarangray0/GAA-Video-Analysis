"""Rendering utilities for the video analysis pipeline."""
import cv2
import numpy as np


def warp_frame(img: np.ndarray, H: np.ndarray, out_w: int, out_h: int) -> np.ndarray:
    """Warp a video frame to the pitch canvas using a homography matrix."""
    return cv2.warpPerspective(img, H, (out_w, out_h))
