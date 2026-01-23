"""Homography computation and pixel-to-pitch mapping."""
from typing import Tuple, Dict, List
import numpy as np
import cv2

from pipeline.config import PITCH_W, PITCH_H, OUT_W, OUT_H, K1
from pipeline.schemas import PitchPoint
from gaa_pitch_config import GAA_PITCH_VERTICES


def compute_homography(
    pts_image: np.ndarray,
    pts_pitch_norm: np.ndarray
) -> np.ndarray:
    """
    Compute homography matrix from image points to normalized pitch points.
    
    Args:
        pts_image: Nx2 array of image coordinates (x_img, y_img)
        pts_pitch_norm: Nx2 array of normalized pitch coordinates (x_pitch_norm, y_pitch_norm)
    
    Returns:
        3x3 homography matrix
    """
    H, _ = cv2.findHomography(
        pts_image.astype(np.float32),
        pts_pitch_norm.astype(np.float32),
        cv2.RANSAC,
        5.0
    )
    
    if H is None:
        raise ValueError("Failed to compute homography")
    
    return H


def normalize_pitch_points(pts_pitch: np.ndarray) -> np.ndarray:
    """
    Normalize pitch points to output canvas dimensions.
    
    Args:
        pts_pitch: Nx2 array of pitch coordinates in meters (x, y)
    
    Returns:
        Nx2 array of normalized pitch coordinates in pixels
    """
    pts_pitch_norm = np.column_stack([
        pts_pitch[:, 0] / PITCH_W * OUT_W,
        pts_pitch[:, 1] / PITCH_H * OUT_H
    ]).astype(np.float32)
    
    return pts_pitch_norm


def map_pixel_to_distorted_pitch(
    x_img: float,
    y_img: float,
    H: np.ndarray,
    out_w: int = None,
    out_h: int = None,
    k1: float = None
) -> Tuple[float, float]:
    """
    Map a single image pixel to distorted pitch coordinates.
    
    Uses homography + radial distortion model from the notebook.
    
    Args:
        x_img: Image x coordinate
        y_img: Image y coordinate
        H: 3x3 homography matrix
        out_w: Output canvas width (defaults to OUT_W)
        out_h: Output canvas height (defaults to OUT_H)
        k1: Radial distortion coefficient (defaults to K1)
    
    Returns:
        Tuple of (x_pitch, y_pitch) in normalized pitch space
    """
    if out_w is None:
        out_w = OUT_W
    if out_h is None:
        out_h = OUT_H
    if k1 is None:
        k1 = K1
    
    H_inv = np.linalg.inv(H)
    
    # Project to pitch plane (ideal)
    p = np.array([x_img, y_img, 1.0], dtype=np.float32)
    pitch = H @ p
    pitch /= pitch[2]
    
    x, y = pitch[0], pitch[1]
    
    # Apply radial distortion in pitch space
    cx, cy = out_w / 2, out_h / 2
    dx = x - cx
    dy = y - cy
    r2 = dx * dx + dy * dy
    
    x_d = x + dx * k1 * r2
    y_d = y + dy * k1 * r2
    
    return x_d, y_d


def compute_homographies_from_annotations(
    annotations: Dict[int, List[PitchPoint]]
) -> Dict[int, np.ndarray]:
    """
    Compute homography matrices for multiple frames from pitch annotations.
    
    Args:
        annotations: Dict mapping frame_idx to list of (pitch_id, x_img, y_img) tuples
    
    Returns:
        Dict mapping frame_idx to 3x3 homography matrix
    """
    homographies = {}
    
    for frame_idx, points in annotations.items():
        if len(points) < 4:
            continue
        
        # Extract image points
        pts_image = np.array([
            [p.x_img, p.y_img] for p in points
        ], dtype=np.float32)
        
        # Extract pitch points from pitch_id
        pts_pitch = np.array([
            GAA_PITCH_VERTICES[p.pitch_id] for p in points
        ], dtype=np.float32)
        
        # Normalize pitch points
        pts_pitch_norm = normalize_pitch_points(pts_pitch)
        
        # Compute homography
        try:
            H = compute_homography(pts_image, pts_pitch_norm)
            homographies[frame_idx] = H
        except ValueError:
            continue
    
    return homographies
