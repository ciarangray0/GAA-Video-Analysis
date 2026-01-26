"""Homography computation and pixel-to-pitch mapping.

Canonical Coordinate System:
============================
This system uses a FIXED TOP-DOWN PITCH CANVAS in PIXELS as the primary
coordinate space. This matches the original research notebook exactly.

    Image pixels (camera) → Homography → Pitch canvas pixels → Distortion → Output

The pitch canvas is a fixed size (e.g., 850 × 1450 pixels) used for:
    - Visualization
    - Analysis
    - Interpolation
    - All downstream processing

IMPORTANT:
- Meters are NOT used in this pipeline
- Radial distortion is ALWAYS applied (not optional)
- All player positions are in pitch canvas pixels
- Interpolation happens in pitch-pixel space

This design prioritizes visual correctness matching the notebook output
over physical/metric correctness.
"""
from typing import Tuple, Dict, List
import numpy as np
import cv2

from pipeline.config import OUT_W, OUT_H, K1
from pipeline.schemas import PitchPoint
from pipeline.gaa_pitch_config import GAA_PITCH_VERTICES

# Pitch canvas dimensions (pixels) - the canonical coordinate space
PITCH_CANVAS_W = OUT_W  # e.g., 850 pixels
PITCH_CANVAS_H = OUT_H  # e.g., 1450 pixels

# Pitch dimensions in meters (only used to compute normalized vertex positions)
PITCH_METERS_W = 85.0
PITCH_METERS_H = 145.0


def _meters_to_canvas_pixels(x_m: float, y_m: float) -> Tuple[float, float]:
    """
    Internal helper: Convert pitch vertex coordinates from meters to canvas pixels.

    This is ONLY used when setting up the homography destination points.
    All other code works in canvas pixels directly.
    """
    x_px = x_m / PITCH_METERS_W * PITCH_CANVAS_W
    y_px = y_m / PITCH_METERS_H * PITCH_CANVAS_H
    return x_px, y_px


def compute_homography(
    pts_image: np.ndarray,
    pts_pitch_canvas: np.ndarray
) -> np.ndarray:
    """
    Compute homography matrix from image points to pitch canvas points.

    Args:
        pts_image: Nx2 array of image coordinates (x_img, y_img) in camera pixels
        pts_pitch_canvas: Nx2 array of pitch canvas coordinates (x, y) in pixels

    Returns:
        3x3 homography matrix that maps: image pixels → pitch canvas pixels
    """
    H, _ = cv2.findHomography(
        pts_image.astype(np.float32),
        pts_pitch_canvas.astype(np.float32),
        cv2.RANSAC,
        5.0
    )
    
    if H is None:
        raise ValueError("Failed to compute homography")
    
    return H


def map_pixel_to_pitch(
    x_img: float,
    y_img: float,
    H: np.ndarray,
    out_w: int = None,
    out_h: int = None,
    k1: float = None
) -> Tuple[float, float]:
    """
    Map an image pixel to pitch canvas coordinates WITH radial distortion.

    This is the PRIMARY mapping function. It reproduces the notebook's
    map_pixel_to_distorted_pitch() exactly:

        1. Apply homography: image pixels → pitch canvas pixels
        2. Apply radial distortion (ALWAYS, not optional)

    The output is the canonical player position used for all visualization
    and analysis.

    Args:
        x_img: Image x coordinate (camera pixels)
        y_img: Image y coordinate (camera pixels)
        H: 3x3 homography matrix (image → pitch canvas)
        out_w: Pitch canvas width (defaults to OUT_W)
        out_h: Pitch canvas height (defaults to OUT_H)
        k1: Radial distortion coefficient (defaults to K1)

    Returns:
        Tuple of (x_pitch, y_pitch) in PITCH CANVAS PIXELS (with distortion)
    """
    if out_w is None:
        out_w = OUT_W
    if out_h is None:
        out_h = OUT_H
    if k1 is None:
        k1 = K1

    # Step 1: Apply homography (image pixels → pitch canvas pixels)
    p = np.array([x_img, y_img, 1.0], dtype=np.float32)
    pitch = H @ p
    pitch /= pitch[2]  # Normalize homogeneous coordinates

    x, y = pitch[0], pitch[1]

    # Step 2: Apply radial distortion (ALWAYS - this is NOT optional)
    # This compensates for perspective effects and matches notebook output
    cx, cy = out_w / 2.0, out_h / 2.0
    dx = x - cx
    dy = y - cy
    r2 = dx * dx + dy * dy

    x_d = x + dx * k1 * r2
    y_d = y + dy * k1 * r2

    return float(x_d), float(y_d)


def compute_homographies_from_annotations(
    annotations: Dict[int, List[PitchPoint]]
) -> Dict[int, np.ndarray]:
    """
    Compute homography matrices for anchor frames from pitch annotations.

    Each homography maps: image pixels → pitch canvas pixels

    The pitch vertices (from GAA_PITCH_VERTICES) are defined in meters,
    but we convert them to canvas pixels for the homography computation.

    Args:
        annotations: Dict mapping frame_idx to list of PitchPoint objects
                    Each PitchPoint has: pitch_id, x_img, y_img

    Returns:
        Dict mapping frame_idx to 3x3 homography matrix
        The homography transforms: image pixels → pitch canvas pixels
    """
    homographies = {}
    
    for frame_idx, points in annotations.items():
        if len(points) < 4:
            continue
        
        # Extract image points (camera pixels)
        pts_image = np.array([
            [p.x_img, p.y_img] for p in points
        ], dtype=np.float32)
        
        # Extract pitch points from pitch_id and convert to canvas pixels
        # GAA_PITCH_VERTICES are in meters, we convert to canvas pixels
        pts_pitch_canvas = np.array([
            _meters_to_canvas_pixels(*GAA_PITCH_VERTICES[p.pitch_id])
            for p in points
        ], dtype=np.float32)
        
        # Compute homography: image pixels → pitch canvas pixels
        try:
            H = compute_homography(pts_image, pts_pitch_canvas)
            homographies[frame_idx] = H
        except ValueError:
            continue
    
    return homographies


# =============================================================================
# LEGACY ALIASES - for backwards compatibility
# =============================================================================

# Alias for the primary mapping function
map_pixel_to_distorted_pitch = map_pixel_to_pitch

def map_pixel_to_pitch_meters(x_img: float, y_img: float, H: np.ndarray) -> Tuple[float, float]:
    """
    DEPRECATED: This system uses pitch canvas pixels, not meters.

    This function is kept for backwards compatibility but should not be used.
    Use map_pixel_to_pitch() instead.
    """
    # Map to canvas pixels first
    x_px, y_px = map_pixel_to_pitch(x_img, y_img, H)
    # Convert canvas pixels to meters (for legacy code only)
    x_m = x_px / PITCH_CANVAS_W * PITCH_METERS_W
    y_m = y_px / PITCH_CANVAS_H * PITCH_METERS_H
    return x_m, y_m

