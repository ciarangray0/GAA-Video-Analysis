"""Homography computation and pixel-to-pitch mapping.

Canonical Coordinate System:
============================
This system uses a FIXED TOP-DOWN PITCH CANVAS in PIXELS as the primary
coordinate space. This matches the original research notebook exactly.

    Image pixels (camera) → Homography → Pitch canvas pixels → Distortion → Output

The pitch canvas is a fixed size (e.g., 850 × 1400 pixels) used for:
    - Visualization
    - Analysis
    - Interpolation
    - All downstream processing

IMPORTANT:
- Meters are NOT used in this pipeline
- Radial distortion is ALWAYS applied
- All player positions are in pitch canvas pixels
- Interpolation happens in pitch-pixel space

This design prioritizes visual correctness over physical/metric correctness.
"""
from typing import Tuple, Dict, List
import numpy as np
import cv2

from pipeline.config import OUT_W, OUT_H, K1
from pipeline.schemas import PitchPoint, LineAnnotation
from pipeline.gaa_pitch_config import GAA_PITCH_VERTICES

# Pitch canvas dimensions (pixels) - the canonical coordinate space
PITCH_CANVAS_W = OUT_W  # e.g., 850 pixels
PITCH_CANVAS_H = OUT_H  # e.g., 1400 pixels

# Pitch dimensions in meters (only used to compute normalized vertex positions)
PITCH_METERS_W = 85.0
PITCH_METERS_H = 140.0


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


def compute_homographies_with_lines(
    annotations: Dict[int, Dict],
    num_samples_per_line: int = 10,
    max_iterations: int = 3,
    keypoint_weight: int = 3
) -> Tuple[Dict[int, np.ndarray], Dict[int, dict]]:
    """
    Compute homography matrices with support for line constraints.

    This is the enhanced version of compute_homographies_from_annotations
    that supports both keypoint and line annotations. Line annotations
    provide additional constraints for regions where point intersections
    are not visible (e.g., midfield).

    Args:
        annotations: Dict mapping frame_idx to annotation dict:
            {
                "keypoints": List[PitchPoint],
                "lines": List[LineAnnotation]  # Optional
            }
        num_samples_per_line: Points to sample per line constraint
        max_iterations: Refinement iterations for line constraints
        keypoint_weight: Weight multiplier for keypoints vs line points

    Returns:
        Tuple of:
        - homographies: Dict mapping frame_idx to 3x3 homography matrix
        - info: Dict mapping frame_idx to computation info dict with:
            - iterations: Number of iterations performed
            - valid_lines: Number of valid line annotations used
            - line_warnings: List of warning messages
            - synthetic_points: Total synthetic points generated
            - converged: Whether algorithm converged
    """
    from pipeline.line_constraints import compute_line_constrained_homography
    from pipeline.schemas import PitchPoint, LineAnnotation

    homographies = {}
    computation_info = {}

    for frame_idx, ann in annotations.items():
        # Handle both old format (list of PitchPoint) and new format (dict)
        if isinstance(ann, list):
            # Old format: list of PitchPoint objects
            keypoints = ann
            lines = []
        else:
            # New format: dict with keypoints and lines
            keypoints = ann.get("keypoints", [])
            lines = ann.get("lines", [])

        if len(keypoints) < 4:
            continue

        # Extract keypoint correspondences
        pts_image = np.array([
            [p.x_img, p.y_img] for p in keypoints
        ], dtype=np.float32)

        pts_canvas = np.array([
            _meters_to_canvas_pixels(*GAA_PITCH_VERTICES[p.pitch_id])
            for p in keypoints
        ], dtype=np.float32)

        # Convert line annotations to dict format for line_constraints module
        line_dicts = []
        for line in lines:
            if isinstance(line, LineAnnotation):
                line_dicts.append({
                    "line_id": line.line_id,
                    "u1": line.u1, "v1": line.v1,
                    "u2": line.u2, "v2": line.v2
                })
            elif isinstance(line, dict):
                line_dicts.append(line)

        try:
            H, info = compute_line_constrained_homography(
                pts_image, pts_canvas,
                line_dicts,
                num_samples_per_line=num_samples_per_line,
                max_iterations=max_iterations,
                keypoint_weight=keypoint_weight
            )
            homographies[frame_idx] = H
            computation_info[frame_idx] = info
        except ValueError as e:
            computation_info[frame_idx] = {
                'error': str(e),
                'valid_lines': 0,
                'iterations': 0
            }
            continue

    return homographies, computation_info


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

