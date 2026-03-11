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
import re
from typing import Tuple, Dict, List
import numpy as np
import cv2

from pipeline.config import OUT_W, OUT_H, K1
from pipeline.schemas import PitchPoint, LineAnnotation
from pipeline.gaa_pitch_config import GAA_PITCH_VERTICES, GAA_PITCH_WIDTH, GAA_PITCH_LENGTH

# Use USAC_MAGSAC if available (OpenCV 4.5+); it handles >50% outlier ratios
# better than standard RANSAC. Fall back gracefully on older OpenCV builds.
_RANSAC_METHOD = getattr(cv2, 'USAC_MAGSAC', cv2.RANSAC)

# Reprojection-error threshold (px) above which an anchor homography is
# considered bad-quality and will be replaced by neighbour interpolation.
_BAD_ANCHOR_THRESHOLD = 30.0

def _meters_to_canvas_pixels(x_m: float, y_m: float) -> Tuple[float, float]:
    """Convert pitch vertex coordinates (meters) to canvas pixels.

    Only used when setting up homography destination points.
    """
    return x_m / GAA_PITCH_WIDTH * OUT_W, y_m / GAA_PITCH_LENGTH * OUT_H


def resolve_pitch_coordinates(pitch_id: str) -> Tuple[float, float]:
    """
    Resolve a pitch_id string to (x_meters, y_meters) coordinates.

    Supports two formats:
    1. Named vertex: pitch_id must be a key in GAA_PITCH_VERTICES.
       Returns the stored (x, y) tuple in meters.
    2. Line point: pitch_id matches ``line_{name}_x{X}_y{Y}`` where X and Y
       are floating-point meter values (e.g. ``line_45m_top_x25.3_y45.0``).
       Returns (float(X), float(Y)).

    Args:
        pitch_id: Vertex name or encoded line-point identifier.

    Returns:
        Tuple of (x_meters, y_meters) in pitch coordinate space (meters).

    Raises:
        ValueError: If pitch_id is not a known vertex name and does not match
                    the ``line_{name}_x{X}_y{Y}`` pattern.
    """
    if pitch_id in GAA_PITCH_VERTICES:
        return GAA_PITCH_VERTICES[pitch_id]

    # Try to parse line_{name}_x{X}_y{Y} format
    match = re.match(r'^line_.+_x([-\d.]+)_y([-\d.]+)$', pitch_id)
    if match:
        return float(match.group(1)), float(match.group(2))

    raise ValueError(
        f"Unrecognized pitch_id: '{pitch_id}'. "
        "Must be a known vertex name or follow the 'line_<name>_x<X>_y<Y>' format."
    )


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
        _RANSAC_METHOD,
        5.0
    )

    # USAC_MAGSAC may return None when given the minimum 4 points (it needs an
    # overdetermined system for its statistical framework).  Fall back to plain
    # RANSAC in that case so the function stays robust to small point sets.
    if H is None and _RANSAC_METHOD != cv2.RANSAC:
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
            _meters_to_canvas_pixels(*resolve_pitch_coordinates(p.pitch_id))
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
            _meters_to_canvas_pixels(*resolve_pitch_coordinates(p.pitch_id))
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
                keypoint_weight=keypoint_weight,
                prefer_line_pts_for_init=True,
                min_line_pts_for_init=4,
            )
            homographies[frame_idx] = H
            # Augment info with keypoint count and reprojection error
            info['num_keypoints'] = len(keypoints)
            try:
                pts_h = np.column_stack([pts_image, np.ones(len(pts_image))])
                projected = (H @ pts_h.T).T
                projected = projected[:, :2] / projected[:, 2:3]
                errors = np.sqrt(((projected - pts_canvas) ** 2).sum(axis=1))
                info['repr_mean'] = round(float(np.mean(errors)), 2)
                info['repr_max'] = round(float(np.max(errors)), 2)
            except Exception:
                pass
            info['quality'] = (
                'good' if info.get('repr_mean', float('inf')) <= _BAD_ANCHOR_THRESHOLD else 'bad'
            )
            computation_info[frame_idx] = info
        except ValueError as e:
            computation_info[frame_idx] = {
                'error': str(e),
                'valid_lines': 0,
                'iterations': 0
            }
            continue

    # Post-processing: replace bad-anchor H matrices with an interpolated H
    # from neighboring good anchors (mean reprojection error > _BAD_ANCHOR_THRESHOLD).
    # This prevents a grossly wrong anchor H from poisoning ORB propagation.
    sorted_frames = sorted(homographies.keys())
    good_frames = [
        f for f in sorted_frames
        if computation_info.get(f, {}).get('repr_mean', float('inf')) <= _BAD_ANCHOR_THRESHOLD
    ]
    if good_frames:
        for f in sorted_frames:
            if computation_info.get(f, {}).get('repr_mean', float('inf')) > _BAD_ANCHOR_THRESHOLD:
                left = [g for g in good_frames if g <= f]
                right = [g for g in good_frames if g >= f]
                if left and right:
                    f_left, f_right = left[-1], right[0]
                    if f_left == f_right:
                        homographies[f] = homographies[f_left]
                    else:
                        alpha = (f - f_left) / (f_right - f_left)
                        homographies[f] = (
                            (1.0 - alpha) * homographies[f_left]
                            + alpha * homographies[f_right]
                        )
                elif left:
                    homographies[f] = homographies[left[-1]]
                elif right:
                    homographies[f] = homographies[right[0]]

    return homographies, computation_info

