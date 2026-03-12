"""Line-based homography constraints for improved stability.

This module provides functions to use annotated pitch lines as additional
constraints when computing homographies. This is particularly useful for
GAA matches where midfield regions have visible horizontal lines (13m, 20m,
45m, 65m, halfway) but no visible point intersections.

Mathematical Insight:
====================
A pitch line (e.g., the 20m line) provides a one-dimensional constraint:
every point on that line in the image has a KNOWN Y-coordinate in world
space, but an UNKNOWN X-coordinate.

By sampling points along the annotated line and using the current homography
estimate to infer X-coordinates, we can generate "synthetic" point
correspondences that improve homography stability in regions far from
visible keypoints.

Algorithm Overview:
==================
1. User annotates 4+ keypoint correspondences (corners, goal posts, etc.)
2. User annotates pitch lines by clicking two points on each visible line
3. Backend computes initial H from keypoints only
4. For each annotated line:
   a. Sample N points along the line in image space
   b. Project through H to estimate X-coordinates
   c. Create synthetic correspondences with known Y, estimated X
5. Re-compute H using keypoints + synthetic points (weighted)
6. Iterate until convergence (usually 2-3 iterations)

Usage:
=====
    from pipeline.line_constraints import compute_line_constrained_homography

    H = compute_line_constrained_homography(
        pts_image_keypoints,      # Nx2 keypoint image coords
        pts_canvas_keypoints,     # Nx2 keypoint canvas coords
        line_annotations,         # List of line dicts
        num_samples_per_line=10,
        max_iterations=3
    )
"""

from typing import List, Tuple, Dict, Optional
from collections import defaultdict
import numpy as np
import cv2

from pipeline.config import OUT_W, OUT_H

# Use USAC_MAGSAC if available (OpenCV 4.5+); it handles >50% outlier ratios
# better than standard RANSAC. Fall back gracefully on older OpenCV builds.
_RANSAC_METHOD = getattr(cv2, 'USAC_MAGSAC', cv2.RANSAC)


# =============================================================================
# GAA Pitch Line Configuration
# =============================================================================

# GAA pitch line Y-values in meters
# These are horizontal lines that cross the full width of the pitch
GAA_PITCH_LINES = {
    # Top half of pitch (near goal at Y=0)
    "endline_top": 0.0,
    "small_rectangle_top": 4.5,      # Goal area line
    "13m_top": 13.0,
    "20m_top": 20.0,
    "45m_top": 45.0,
    "65m_top": 65.0,

    # Halfway line (pitch is 140m, halfway at 70m)
    "halfway": 70.0,

    # Bottom half of pitch (near goal at Y=140m)
    "65m_bottom": 75.0,              # 140 - 65 = 75
    "45m_bottom": 95.0,              # 140 - 45 = 95
    "20m_bottom": 120.0,             # 140 - 20 = 120
    "13m_bottom": 127.0,             # 140 - 13 = 127
    "small_rectangle_bottom": 135.5, # 140 - 4.5 = 135.5
    "endline_bottom": 140.0,
}

# GAA pitch sideline X-values in meters (vertical lines running full pitch length)
GAA_PITCH_SIDELINES = {
    "left_sideline":   0.0,   # x = 0m (left boundary)
    "right_sideline":  85.0,  # x = 85m (right boundary)
    "13m_box_left":    33.0,  # x = 33m (left 13m box side)
    "13m_box_right":   52.0,  # x = 52m (right 13m box side)
    "small_arc_left":  29.5,  # x = 29.5m (left small arc side)
    "small_arc_right": 55.5,  # x = 55.5m (right small arc side)
}

# Pitch dimensions
PITCH_METERS_H = 140.0  # Total pitch height in meters
PITCH_METERS_W = 85.0   # Total pitch width in meters


def get_line_y_canvas(line_id: str) -> float:
    """
    Get the Y coordinate in canvas pixels for a line ID.

    Args:
        line_id: Line identifier (e.g., "20m_top", "halfway")

    Returns:
        Y coordinate in pitch canvas pixels

    Raises:
        ValueError: If line_id is not recognized
    """
    if line_id not in GAA_PITCH_LINES:
        raise ValueError(
            f"Unknown line ID: {line_id}. "
            f"Valid options: {list(GAA_PITCH_LINES.keys())}"
        )
    y_meters = GAA_PITCH_LINES[line_id]
    # Convert to canvas pixels (OUT_H pixels for PITCH_METERS_H meters)
    return y_meters / PITCH_METERS_H * OUT_H


def get_sideline_x_canvas(line_id: str) -> float:
    """
    Get the X coordinate in canvas pixels for a sideline ID.

    Args:
        line_id: Sideline identifier (e.g., "left_sideline", "right_sideline")

    Returns:
        X coordinate in pitch canvas pixels

    Raises:
        ValueError: If line_id is not a recognised sideline
    """
    if line_id not in GAA_PITCH_SIDELINES:
        raise ValueError(
            f"Unknown sideline ID: {line_id}. "
            f"Valid options: {list(GAA_PITCH_SIDELINES.keys())}"
        )
    x_meters = GAA_PITCH_SIDELINES[line_id]
    # Convert to canvas pixels (OUT_W pixels for PITCH_METERS_W meters)
    return x_meters / PITCH_METERS_W * OUT_W


def get_available_lines() -> Dict[str, float]:
    """Return dict of available line IDs and their Y values in meters."""
    return GAA_PITCH_LINES.copy()


# =============================================================================
# Point Sampling
# =============================================================================

def sample_points_on_line(
    u1: float, v1: float,
    u2: float, v2: float,
    num_samples: int = 10
) -> np.ndarray:
    """
    Sample N points uniformly along a line segment in image space.

    Args:
        u1, v1: First endpoint in image pixels
        u2, v2: Second endpoint in image pixels
        num_samples: Number of points to sample (including endpoints)

    Returns:
        Nx2 array of image points [(u, v), ...]
    """
    if num_samples < 2:
        num_samples = 2

    t_values = np.linspace(0.0, 1.0, num_samples)
    u_samples = (1 - t_values) * u1 + t_values * u2
    v_samples = (1 - t_values) * v1 + t_values * v2
    return np.column_stack([u_samples, v_samples]).astype(np.float32)


def get_point_weights(num_samples: int) -> np.ndarray:
    """
    Generate confidence weights for sampled points.

    Points near the center of the line segment are weighted higher
    than endpoints, as they're more reliable (less affected by
    annotation precision at endpoints).

    Args:
        num_samples: Number of points

    Returns:
        Array of weights in [0.5, 1.0]
    """
    if num_samples == 1:
        return np.array([1.0])

    t_values = np.linspace(0.0, 1.0, num_samples)
    # Parabolic falloff: max at t=0.5, min at t=0 and t=1
    # Weight = 0.5 + 0.5 * (1 - (2t - 1)^2)
    weights = 0.5 + 0.5 * (1 - (2 * t_values - 1) ** 2)
    return weights.astype(np.float32)


# =============================================================================
# Synthetic Correspondence Generation
# =============================================================================

def generate_synthetic_correspondences(
    line_annotation: dict,
    H_current: np.ndarray,
    num_samples: int = 10,
    clamp_x: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate synthetic point correspondences from a line annotation.

    The key insight: we know Y_world exactly (from the line ID),
    and estimate X_world by projecting through the current homography.

    Args:
        line_annotation: Dict with keys:
            - line_id: str (e.g., "20m_top")
            - u1, v1: First point in image pixels
            - u2, v2: Second point in image pixels
        H_current: Current homography estimate (3x3 matrix)
        num_samples: Number of points to sample along line
        clamp_x: Whether to clamp X to valid pitch range

    Returns:
        Tuple of:
        - pts_image: Nx2 array of image points
        - pts_world: Nx2 array of world points (canvas pixels)
        - weights: N array of confidence weights

    Raises:
        ValueError: If line_id is not recognized
        ValueError: If H_current is singular or invalid
    """
    # Get fixed Y coordinate for this line
    y_canvas_fixed = get_line_y_canvas(line_annotation['line_id'])

    # Sample points along the line in image space
    pts_image = sample_points_on_line(
        line_annotation['u1'], line_annotation['v1'],
        line_annotation['u2'], line_annotation['v2'],
        num_samples
    )

    # Project through current homography to estimate X coordinates
    # pts_image is Nx2, we need to make it homogeneous Nx3
    pts_homogeneous = np.hstack([pts_image, np.ones((len(pts_image), 1))])

    # Apply homography: H @ [u, v, 1]^T for each point
    pts_projected = (H_current @ pts_homogeneous.T).T

    # Normalize homogeneous coordinates
    w = pts_projected[:, 2:3]
    if np.any(np.abs(w) < 1e-10):
        raise ValueError("Homography projects some points to infinity")
    pts_projected = pts_projected[:, :2] / w

    # Extract estimated X coordinates
    x_estimated = pts_projected[:, 0]

    # Optionally clamp X to valid pitch range
    if clamp_x:
        x_estimated = np.clip(x_estimated, 0, OUT_W)

    # Create world points: use estimated X, but FIXED Y
    pts_world = np.column_stack([
        x_estimated,
        np.full(len(x_estimated), y_canvas_fixed)
    ]).astype(np.float32)

    # Generate confidence weights
    weights = get_point_weights(num_samples)

    return pts_image, pts_world, weights


def generate_synthetic_correspondences_vertical(
    line_annotation: dict,
    H_current: np.ndarray,
    num_samples: int = 10,
    clamp_y: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate synthetic point correspondences from a vertical sideline annotation.

    The key insight: we know X_world exactly (from the sideline ID),
    and estimate Y_world by projecting through the current homography.

    Args:
        line_annotation: Dict with keys:
            - line_id: str (e.g., "left_sideline", "right_sideline")
            - u1, v1: First point in image pixels
            - u2, v2: Second point in image pixels
        H_current: Current homography estimate (3x3 matrix)
        num_samples: Number of points to sample along line
        clamp_y: Whether to clamp Y to valid pitch range

    Returns:
        Tuple of:
        - pts_image: Nx2 array of image points
        - pts_world: Nx2 array of world points (canvas pixels)
        - weights: N array of confidence weights

    Raises:
        ValueError: If line_id is not a recognised sideline
        ValueError: If H_current is singular or invalid
    """
    # Get fixed X coordinate for this sideline
    x_canvas_fixed = get_sideline_x_canvas(line_annotation['line_id'])

    # Sample points along the line in image space
    pts_image = sample_points_on_line(
        line_annotation['u1'], line_annotation['v1'],
        line_annotation['u2'], line_annotation['v2'],
        num_samples
    )

    # Project through current homography to estimate Y coordinates
    pts_homogeneous = np.hstack([pts_image, np.ones((len(pts_image), 1))])
    pts_projected = (H_current @ pts_homogeneous.T).T

    # Normalize homogeneous coordinates
    w = pts_projected[:, 2:3]
    if np.any(np.abs(w) < 1e-10):
        raise ValueError("Homography projects some points to infinity")
    pts_projected = pts_projected[:, :2] / w

    # Extract estimated Y coordinates
    y_estimated = pts_projected[:, 1]

    # Optionally clamp Y to valid pitch range
    if clamp_y:
        y_estimated = np.clip(y_estimated, 0, OUT_H)

    # Create world points: use FIXED X, estimated Y
    pts_world = np.column_stack([
        np.full(len(y_estimated), x_canvas_fixed),
        y_estimated
    ]).astype(np.float32)

    # Generate confidence weights
    weights = get_point_weights(num_samples)

    return pts_image, pts_world, weights


# =============================================================================
# Line Validation
# =============================================================================

def validate_line_annotation(
    line_annotation: dict,
    H_initial: np.ndarray,
    y_tolerance_pixels: float = 100.0
) -> Tuple[bool, str]:
    """
    Validate a line annotation for geometric consistency.

    For horizontal lines: checks that the average projected Y is close to
    the expected Y for that line.
    For vertical sidelines: checks that the average projected X is close to
    the expected X for that sideline.
    """
    line_id = line_annotation['line_id']
    is_vertical = line_id in GAA_PITCH_SIDELINES

    if is_vertical:
        try:
            x_expected = get_sideline_x_canvas(line_id)
        except ValueError as e:
            return False, str(e)
    else:
        try:
            y_expected = get_line_y_canvas(line_id)
        except ValueError as e:
            return False, str(e)

    p1 = np.array([line_annotation['u1'], line_annotation['v1'], 1.0])
    p2 = np.array([line_annotation['u2'], line_annotation['v2'], 1.0])

    proj1 = H_initial @ p1
    proj2 = H_initial @ p2

    if abs(proj1[2]) < 1e-10 or abs(proj2[2]) < 1e-10:
        return False, "Line endpoints project to infinity"

    if is_vertical:
        x1 = proj1[0] / proj1[2]
        x2 = proj2[0] / proj2[2]
        x_avg = (x1 + x2) / 2
        x_error = abs(x_avg - x_expected)
        if x_error > y_tolerance_pixels * 1.5:
            return False, (
                f"Projected X ({x_avg:.1f}px) is far from expected "
                f"({x_expected:.1f}px) for sideline '{line_id}'. "
                f"Error: {x_error:.1f}px"
            )
    else:
        y1 = proj1[1] / proj1[2]
        y2 = proj2[1] / proj2[2]
        # Only check average Y proximity to expected — not endpoint spread
        y_avg = (y1 + y2) / 2
        y_error = abs(y_avg - y_expected)
        if y_error > y_tolerance_pixels * 1.5:
            return False, (
                f"Projected Y ({y_avg:.1f}px) is far from expected "
                f"({y_expected:.1f}px) for line '{line_id}'. "
                f"Error: {y_error:.1f}px"
            )

    return True, ""


def filter_valid_line_annotations(
    line_annotations: List[dict],
    H_initial: np.ndarray,
    y_tolerance_pixels: float = 100.0
) -> Tuple[List[dict], List[str]]:
    """
    Filter line annotations, keeping only those that pass validation.

    Args:
        line_annotations: List of line annotation dicts
        H_initial: Initial homography estimate
        y_tolerance_pixels: Tolerance for validation

    Returns:
        Tuple of (valid_annotations, warning_messages)
    """
    valid = []
    warnings = []

    for line_ann in line_annotations:
        is_valid, error = validate_line_annotation(
            line_ann, H_initial, y_tolerance_pixels
        )
        if is_valid:
            valid.append(line_ann)
        else:
            warnings.append(f"Skipping line '{line_ann.get('line_id', 'unknown')}': {error}")

    return valid, warnings


# =============================================================================
# Main Homography Computation
# =============================================================================

def compute_line_constrained_homography(
    pts_image_keypoints: np.ndarray,
    pts_canvas_keypoints: np.ndarray,
    line_annotations: List[dict],
    num_samples_per_line: int = 10,
    max_iterations: int = 3,
    keypoint_weight: int = 3,
    validate_lines: bool = True,
    ransac_threshold: float = 5.0,
    prefer_line_pts_for_init: bool = True,
    min_line_pts_for_init: int = 4,
) -> Tuple[np.ndarray, dict]:
    """
    Compute homography using both keypoint and line constraints.

    This is the main function for line-constrained homography estimation.

    Algorithm:
    1. Compute initial H from keypoints only
    2. Validate line annotations against initial H
    3. For each iteration:
       a. Generate synthetic points from valid lines using current H
       b. Combine with keypoints (keypoints weighted higher)
       c. Re-compute H using all points
       d. Check for convergence
    4. Return refined H

    Args:
        pts_image_keypoints: Nx2 array of keypoint image coordinates
        pts_canvas_keypoints: Nx2 array of keypoint canvas coordinates
        line_annotations: List of line annotation dicts, each with:
            - line_id: str
            - u1, v1: First point (image pixels)
            - u2, v2: Second point (image pixels)
        num_samples_per_line: Points to sample per line (default: 10)
        max_iterations: Maximum refinement iterations (default: 3)
        keypoint_weight: How many times to weight keypoints vs line points
        validate_lines: Whether to validate line annotations
        ransac_threshold: RANSAC reprojection threshold

    Returns:
        Tuple of:
        - H: 3x3 homography matrix (image pixels → canvas pixels)
        - info: Dict with metadata about the computation:
            - iterations: Number of iterations performed
            - valid_lines: Number of valid line annotations used
            - line_warnings: List of warning messages for invalid lines
            - synthetic_points: Total synthetic points generated
            - converged: Whether the algorithm converged

    Raises:
        ValueError: If fewer than 4 keypoints provided
        ValueError: If homography computation fails
    """
    # Validate inputs
    if len(pts_image_keypoints) < 4:
        raise ValueError(
            f"Need at least 4 keypoints for homography, got {len(pts_image_keypoints)}"
        )

    if pts_image_keypoints.shape != pts_canvas_keypoints.shape:
        raise ValueError(
            f"Keypoint arrays must have same shape: "
            f"{pts_image_keypoints.shape} vs {pts_canvas_keypoints.shape}"
        )

    # Ensure float32 for OpenCV
    pts_image_keypoints = pts_image_keypoints.astype(np.float32)
    pts_canvas_keypoints = pts_canvas_keypoints.astype(np.float32)

    # Initialize info dict
    info = {
        'iterations': 0,
        'valid_lines': 0,
        'line_warnings': [],
        'synthetic_points': 0,
        'converged': False,
        'used_line_pts_for_init': False,
    }

    # ── Step 1: compute initial H ─────────────────────────────────────────────
    H_current = None

    if prefer_line_pts_for_init:
        known_line_ys = set(
            round(y_m / 140.0 * 1400)
            for y_m in GAA_PITCH_LINES.values()
        )
        line_pt_mask = np.zeros(len(pts_canvas_keypoints), dtype=bool)
        for i, (cx, cy) in enumerate(pts_canvas_keypoints):
            for known_y in known_line_ys:
                if abs(cy - known_y) < 3.0:
                    line_pt_mask[i] = True
                    break

        line_pts_img    = pts_image_keypoints[line_pt_mask]
        line_pts_canvas = pts_canvas_keypoints[line_pt_mask]

        if len(line_pts_img) >= min_line_pts_for_init:
            H_init, mask_init = cv2.findHomography(
                line_pts_img, line_pts_canvas, _RANSAC_METHOD, ransac_threshold
            )
            if H_init is not None and mask_init is not None:
                n_inliers = int(mask_init.sum())
                if n_inliers >= 4:
                    H_current = H_init
                    info['used_line_pts_for_init'] = True
                    info['line_warnings'].append(
                        f"Initial H computed from {n_inliers}/{len(line_pts_img)} "
                        f"line_ exact points (prefer_line_pts_for_init=True)"
                    )

    # Fall back to all keypoints if line_ init failed or wasn't attempted
    if H_current is None:
        H_current, mask = cv2.findHomography(
            pts_image_keypoints,
            pts_canvas_keypoints,
            _RANSAC_METHOD,
            ransac_threshold
        )
        # USAC_MAGSAC may return None for small point sets; fall back to RANSAC
        if H_current is None and _RANSAC_METHOD != cv2.RANSAC:
            H_current, mask = cv2.findHomography(
                pts_image_keypoints,
                pts_canvas_keypoints,
                cv2.RANSAC,
                ransac_threshold
            )
        if H_current is None:
            raise ValueError("Failed to compute initial homography from keypoints")

    # ── Step 1b: Iterative reprojection-based outlier removal ─────────────────
    # Remove keypoints whose reprojection error exceeds 25 px (up to 3 passes).
    # This makes the initial H robust when >50% of keypoints are outliers.
    pts_img_clean = pts_image_keypoints.copy()
    pts_can_clean = pts_canvas_keypoints.copy()
    for _outlier_iter in range(3):
        if len(pts_img_clean) < 4:
            break
        pts_h = np.column_stack([pts_img_clean, np.ones(len(pts_img_clean))])
        projected = (H_current @ pts_h.T).T
        projected = projected[:, :2] / projected[:, 2:3]
        errors = np.sqrt(((projected - pts_can_clean) ** 2).sum(axis=1))
        inlier_mask = errors < 25.0
        n_inliers = int(inlier_mask.sum())
        if n_inliers == len(pts_img_clean):
            break  # No outliers remain
        if n_inliers < 4:
            break  # Would remove too many; keep current set
        pts_img_clean = pts_img_clean[inlier_mask]
        pts_can_clean = pts_can_clean[inlier_mask]
        H_refined, _ = cv2.findHomography(
            pts_img_clean, pts_can_clean, _RANSAC_METHOD, ransac_threshold
        )
        if H_refined is not None:
            H_current = H_refined
        else:
            break
    # Use the filtered keypoint set for subsequent line-constraint refinement
    pts_image_keypoints = pts_img_clean
    pts_canvas_keypoints = pts_can_clean

    # If no line annotations, return initial homography
    if not line_annotations:
        return H_current, info

    # Step 2: Validate line annotations
    if validate_lines:
        valid_lines, warnings = filter_valid_line_annotations(
            line_annotations, H_current
        )
        info['line_warnings'] = warnings
    else:
        valid_lines = line_annotations

    info['valid_lines'] = len(valid_lines)

    # If no valid lines, return initial homography
    if not valid_lines:
        return H_current, info

    # Step 3: Iterative refinement with line constraints
    for iteration in range(max_iterations):
        info['iterations'] = iteration + 1

        # Collect synthetic points from all valid lines
        all_pts_image_synthetic = []
        all_pts_world_synthetic = []
        all_weights_synthetic = []

        for line_ann in valid_lines:
            try:
                if line_ann.get('line_id') in GAA_PITCH_SIDELINES:
                    pts_img, pts_world, weights = generate_synthetic_correspondences_vertical(
                        line_ann, H_current, num_samples_per_line
                    )
                else:
                    pts_img, pts_world, weights = generate_synthetic_correspondences(
                        line_ann, H_current, num_samples_per_line
                    )
                all_pts_image_synthetic.append(pts_img)
                all_pts_world_synthetic.append(pts_world)
                all_weights_synthetic.append(weights)
            except ValueError as e:
                # Skip lines that cause projection issues
                info['line_warnings'].append(
                    f"Iteration {iteration}: Skipping line "
                    f"'{line_ann.get('line_id', 'unknown')}': {e}"
                )
                continue

        # If all lines failed, stop iterating
        if not all_pts_image_synthetic:
            break

        # Combine synthetic points
        pts_image_synthetic = np.vstack(all_pts_image_synthetic)
        pts_world_synthetic = np.vstack(all_pts_world_synthetic)

        info['synthetic_points'] = len(pts_image_synthetic)

        # When initial H came from line_ points, reduce keypoint weight
        # so noisy named vertices don't pull H back toward the wrong solution
        effective_weight = 1 if info['used_line_pts_for_init'] else keypoint_weight
        pts_image_weighted  = np.vstack([pts_image_keypoints]  * effective_weight)
        pts_canvas_weighted = np.vstack([pts_canvas_keypoints] * effective_weight)

        # Combine all points: weighted keypoints + synthetic line points
        all_pts_image = np.vstack([pts_image_weighted, pts_image_synthetic])
        all_pts_world = np.vstack([pts_canvas_weighted, pts_world_synthetic])

        # Re-compute homography with all points
        H_new, _ = cv2.findHomography(
            all_pts_image.astype(np.float32),
            all_pts_world.astype(np.float32),
            _RANSAC_METHOD,
            ransac_threshold
        )

        if H_new is None:
            # Refinement failed, keep previous H
            info['line_warnings'].append(
                f"Iteration {iteration}: Homography refinement failed, using previous"
            )
            break

        # Check for convergence using Frobenius norm of difference
        diff = np.linalg.norm(H_new - H_current, ord='fro')

        # Update current homography
        H_current = H_new

        # Check if converged
        if diff < 0.01:
            info['converged'] = True
            break

    return H_current, info


# =============================================================================
# Convenience Functions
# =============================================================================

def compute_initial_homography(
    pts_image: np.ndarray,
    pts_canvas: np.ndarray,
    ransac_threshold: float = 5.0
) -> np.ndarray:
    """
    Compute homography from keypoints only (no line constraints).

    This is a thin wrapper around cv2.findHomography for consistency.

    Args:
        pts_image: Nx2 image coordinates
        pts_canvas: Nx2 canvas coordinates
        ransac_threshold: RANSAC threshold

    Returns:
        3x3 homography matrix
    """
    H, _ = cv2.findHomography(
        pts_image.astype(np.float32),
        pts_canvas.astype(np.float32),
        _RANSAC_METHOD,
        ransac_threshold
    )
    # USAC_MAGSAC may return None for small point sets; fall back to RANSAC
    if H is None and _RANSAC_METHOD != cv2.RANSAC:
        H, _ = cv2.findHomography(
            pts_image.astype(np.float32),
            pts_canvas.astype(np.float32),
            cv2.RANSAC,
            ransac_threshold
        )
    if H is None:
        raise ValueError("Failed to compute homography")
    return H


def preview_synthetic_points(
    line_annotations: List[dict],
    H: np.ndarray,
    num_samples: int = 10
) -> List[dict]:
    """
    Preview the synthetic points that would be generated from line annotations.

    Useful for visualization in the frontend.

    Args:
        line_annotations: List of line annotation dicts
        H: Current homography estimate
        num_samples: Points per line

    Returns:
        List of dicts with 'image_point', 'world_point', 'line_id', 'weight'
    """
    results = []

    for line_ann in line_annotations:
        try:
            if line_ann.get('line_id') in GAA_PITCH_SIDELINES:
                pts_img, pts_world, weights = generate_synthetic_correspondences_vertical(
                    line_ann, H, num_samples
                )
            else:
                pts_img, pts_world, weights = generate_synthetic_correspondences(
                    line_ann, H, num_samples
                )
            for i in range(len(pts_img)):
                results.append({
                    'image_point': (float(pts_img[i, 0]), float(pts_img[i, 1])),
                    'world_point': (float(pts_world[i, 0]), float(pts_world[i, 1])),
                    'line_id': line_ann['line_id'],
                    'weight': float(weights[i])
                })
        except ValueError:
            continue

    return results


# =============================================================================
# DLT-based homography with genuine line constraints (v3)
# =============================================================================

def _hartley_normalise(pts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute Hartley normalisation for a set of 2-D points.

    Returns (T, pts_normalised) where T is the 3×3 transform such that
    T @ [x, y, 1]^T gives the normalised point.  Normalised points have
    their centroid at the origin and a mean distance of sqrt(2) from it.
    """
    cx, cy = float(pts[:, 0].mean()), float(pts[:, 1].mean())
    dists = np.sqrt((pts[:, 0] - cx) ** 2 + (pts[:, 1] - cy) ** 2)
    mean_dist = float(dists.mean())
    if mean_dist < 1e-8:
        mean_dist = 1.0
    scale = np.sqrt(2.0) / mean_dist
    T = np.array([
        [scale, 0.0,   -scale * cx],
        [0.0,   scale, -scale * cy],
        [0.0,   0.0,    1.0       ],
    ], dtype=np.float64)
    pts_h = np.column_stack([pts, np.ones(len(pts))])
    pts_n = (T @ pts_h.T).T[:, :2]
    return T, pts_n


def _dlt_solve(A: np.ndarray) -> np.ndarray:
    """Solve the DLT system Ah = 0 via SVD.

    Returns the 3×3 homography (last right-singular vector, reshaped and
    normalised so H[2,2] = 1).
    """
    _, _, Vt = np.linalg.svd(A, full_matrices=True)
    h = Vt[-1]
    H = h.reshape(3, 3)
    if abs(H[2, 2]) > 1e-10:
        H = H / H[2, 2]
    return H


def _build_dlt_rows(
    kp_img_n: np.ndarray,        # (K, 2) normalised image keypoints
    kp_can_n: np.ndarray,        # (K, 2) normalised canvas keypoints
    h_img_n: List[Tuple],        # [(u_n, v_n), ...] horizontal line samples
    h_y_n: List[float],          # normalised canvas Y for each h sample
    v_img_n: List[Tuple],        # [(u_n, v_n), ...] vertical line samples
    v_x_n: List[float],          # normalised canvas X for each v sample
    keypoint_weight: float,
) -> np.ndarray:
    """Build the coefficient matrix A for the DLT system Ah = 0.

    Row ordering of h is [h11, h12, h13, h21, h22, h23, h31, h32, h33].

    Full keypoint   → 2 rows (X-constraint + Y-constraint), both scaled by
                      *keypoint_weight*.
    Horizontal line → 1 row  (Y-constraint only, X is free).
    Vertical line   → 1 row  (X-constraint only, Y is free).
    """
    rows = []
    n_kp = len(kp_img_n)

    for i in range(n_kp):
        u, v = kp_img_n[i, 0], kp_img_n[i, 1]
        x, y = kp_can_n[i, 0], kp_can_n[i, 1]
        row_x = np.array([-u, -v, -1,  0,  0,  0,  x*u, x*v, x],  dtype=np.float64) * keypoint_weight
        row_y = np.array([ 0,  0,  0, -u, -v, -1,  y*u, y*v, y],  dtype=np.float64) * keypoint_weight
        rows.append(row_x)
        rows.append(row_y)

    for (u, v), y_n in zip(h_img_n, h_y_n):
        rows.append(np.array([0, 0, 0, -u, -v, -1, y_n*u, y_n*v, y_n], dtype=np.float64))

    for (u, v), x_n in zip(v_img_n, v_x_n):
        rows.append(np.array([-u, -v, -1, 0, 0, 0, x_n*u, x_n*v, x_n], dtype=np.float64))

    if not rows:
        raise ValueError("No DLT rows — no constraints provided.")
    return np.array(rows, dtype=np.float64)


def compute_homography_dlt_with_lines(
    keypoints_image: np.ndarray,
    keypoints_canvas: np.ndarray,
    line_annotations: List[dict],
    num_samples_per_line: int = 10,
    ransac_iterations: int = 2000,
    ransac_threshold: float = 10.0,
    keypoint_weight: float = 3.0,
) -> Tuple[np.ndarray, dict]:
    """Compute homography via Direct Linear Transform using line constraints.

    Unlike ``compute_line_constrained_homography``, this function encodes each
    line annotation as a genuine *one-dimensional* DLT constraint rather than
    a synthetic full-point correspondence that circularly depends on the
    current H.

    Mathematical basis
    ------------------
    A horizontal line at known canvas Y gives one DLT row per sample point::

        [0, 0, 0, -u, -v, -1,  Y·u,  Y·v, Y]  ·  h  = 0

    A vertical line at known canvas X gives one DLT row per sample point::

        [-u, -v, -1,  0,  0,  0,  X·u,  X·v, X]  ·  h  = 0

    A full keypoint at (X, Y) gives *both* rows.

    All coordinates are Hartley-normalised before the SVD solve and
    denormalised afterwards.  RANSAC on top removes annotation noise.

    Args:
        keypoints_image:    Nx2 image-space keypoint coordinates (u, v).
        keypoints_canvas:   Nx2 canvas keypoint coordinates (x, y).
        line_annotations:   List of dicts with keys ``line_id``, ``u1``,
                            ``v1``, ``u2``, ``v2``.
        num_samples_per_line: Points sampled along each line segment.
        ransac_iterations:  Number of RANSAC trials.
        ransac_threshold:   Inlier distance threshold in canvas pixels.
                            For keypoints: Euclidean error (both dims).
                            For line samples: error in the constrained
                            dimension only.
        keypoint_weight:    Row-weight multiplier for full-keypoint rows.

    Returns:
        ``(H, info)`` where *H* is the 3×3 homography and *info* is a dict
        with keys ``num_keypoints``, ``num_line_points``, ``num_lines``,
        ``num_inliers``, ``converged``, ``repr_errors_keypoints``,
        ``repr_errors_lines``, ``line_warnings``.

    Raises:
        ValueError: Fewer than 4 total equations.
    """
    keypoints_image  = np.asarray(keypoints_image,  dtype=np.float64)
    keypoints_canvas = np.asarray(keypoints_canvas, dtype=np.float64)
    n_kp = len(keypoints_image)
    line_warnings: List[str] = []

    # ── 1. Sample points from each line annotation ────────────────────────────
    h_img_raw: List[Tuple[float, float]] = []   # (u, v) image coords
    h_y_raw:   List[float]               = []   # known canvas Y (pixels)
    h_line_ids: List[str]                = []

    v_img_raw: List[Tuple[float, float]] = []   # (u, v) image coords
    v_x_raw:   List[float]               = []   # known canvas X (pixels)
    v_line_ids: List[str]                = []

    for ann in line_annotations:
        lid = ann['line_id']
        pts = sample_points_on_line(
            ann['u1'], ann['v1'], ann['u2'], ann['v2'], num_samples_per_line
        )
        if lid in GAA_PITCH_LINES:
            try:
                y_canvas = get_line_y_canvas(lid)
            except ValueError as exc:
                line_warnings.append(str(exc))
                continue
            for p in pts:
                h_img_raw.append((float(p[0]), float(p[1])))
                h_y_raw.append(y_canvas)
                h_line_ids.append(lid)
        elif lid in GAA_PITCH_SIDELINES:
            try:
                x_canvas = get_sideline_x_canvas(lid)
            except ValueError as exc:
                line_warnings.append(str(exc))
                continue
            for p in pts:
                v_img_raw.append((float(p[0]), float(p[1])))
                v_x_raw.append(x_canvas)
                v_line_ids.append(lid)
        else:
            line_warnings.append(f"Unknown line_id '{lid}' — skipped")

    n_h = len(h_img_raw)
    n_v = len(v_img_raw)
    n_eq = 2 * n_kp + n_h + n_v

    if n_eq < 8:
        line_warnings.append(
            f"Only {n_eq} equations available (need ≥8 for a determined system). "
            "Attempting least-norm DLT solve."
        )
    if n_eq < 4:
        raise ValueError(
            f"Too few constraints: {n_eq} equations from {n_kp} keypoints, "
            f"{n_h} horizontal and {n_v} vertical line samples."
        )

    # ── 2. Hartley normalisation ──────────────────────────────────────────────
    # Image: normalise all image points together.
    all_img_pts_list = list(keypoints_image) if n_kp > 0 else []
    if h_img_raw:
        all_img_pts_list += [list(p) for p in h_img_raw]
    if v_img_raw:
        all_img_pts_list += [list(p) for p in v_img_raw]
    all_img_pts = np.array(all_img_pts_list, dtype=np.float64)
    T_img, _ = _hartley_normalise(all_img_pts)

    # Canvas: normalise from keypoints when available; fall back to canvas size.
    if n_kp > 0:
        T_can, kp_can_n = _hartley_normalise(keypoints_canvas)
    else:
        cx_can, cy_can = OUT_W / 2.0, OUT_H / 2.0
        scale_can = np.sqrt(2.0) / np.sqrt(cx_can ** 2 + cy_can ** 2)
        T_can = np.array([
            [scale_can, 0.0,        -scale_can * cx_can],
            [0.0,       scale_can,  -scale_can * cy_can],
            [0.0,       0.0,         1.0               ],
        ], dtype=np.float64)
        kp_can_n = np.empty((0, 2), dtype=np.float64)

    T_can_inv = np.linalg.inv(T_can)

    def _n_img(u: float, v: float) -> Tuple[float, float]:
        p = T_img @ np.array([u, v, 1.0])
        return float(p[0]), float(p[1])

    def _n_can_y(y: float) -> float:
        return float(T_can[1, 1] * y + T_can[1, 2])

    def _n_can_x(x: float) -> float:
        return float(T_can[0, 0] * x + T_can[0, 2])

    kp_img_n = np.array([_n_img(u, v) for u, v in keypoints_image], dtype=np.float64) if n_kp > 0 else np.empty((0, 2))
    h_img_n  = [_n_img(u, v) for u, v in h_img_raw]
    h_y_n    = [_n_can_y(y)  for y   in h_y_raw   ]
    v_img_n  = [_n_img(u, v) for u, v in v_img_raw ]
    v_x_n    = [_n_can_x(x)  for x   in v_x_raw   ]

    # ── 3. Helper: solve DLT for a subset and denormalise ─────────────────────
    def _solve_subset(
        kp_i: np.ndarray, kp_c: np.ndarray,
        hi: List, hy: List, vi: List, vx: List,
    ) -> Optional[np.ndarray]:
        try:
            A = _build_dlt_rows(kp_i, kp_c, hi, hy, vi, vx, keypoint_weight)
            H_n = _dlt_solve(A)
            H = T_can_inv @ H_n @ T_img
            if abs(H[2, 2]) > 1e-10:
                H = H / H[2, 2]
            return H
        except Exception:
            return None

    # ── 4. Base DLT solve using all data ─────────────────────────────────────
    H_base = _solve_subset(kp_img_n, kp_can_n, h_img_n, h_y_n, v_img_n, v_x_n)
    if H_base is None:
        raise ValueError("Base DLT solve failed (SVD error or degenerate input).")

    # ── 5. Inlier scoring helper ──────────────────────────────────────────────
    def _score(H_cand: np.ndarray):
        """Return (kp_mask, h_mask, v_mask, total_inliers) for un-normalised coords."""
        kp_mask = np.zeros(n_kp, dtype=bool)
        for i, (u, v) in enumerate(keypoints_image):
            proj = H_cand @ np.array([u, v, 1.0])
            if abs(proj[2]) < 1e-10:
                continue
            proj = proj / proj[2]
            err = np.sqrt((proj[0] - keypoints_canvas[i, 0]) ** 2 +
                          (proj[1] - keypoints_canvas[i, 1]) ** 2)
            kp_mask[i] = err < ransac_threshold

        h_mask = np.zeros(n_h, dtype=bool)
        for i, ((u, v), y_known) in enumerate(zip(h_img_raw, h_y_raw)):
            proj = H_cand @ np.array([u, v, 1.0])
            if abs(proj[2]) < 1e-10:
                continue
            proj = proj / proj[2]
            h_mask[i] = abs(proj[1] - y_known) < ransac_threshold

        v_mask = np.zeros(n_v, dtype=bool)
        for i, ((u, v), x_known) in enumerate(zip(v_img_raw, v_x_raw)):
            proj = H_cand @ np.array([u, v, 1.0])
            if abs(proj[2]) < 1e-10:
                continue
            proj = proj / proj[2]
            v_mask[i] = abs(proj[0] - x_known) < ransac_threshold

        total = int(kp_mask.sum()) + int(h_mask.sum()) + int(v_mask.sum())
        return kp_mask, h_mask, v_mask, total

    # ── 6. RANSAC loop ────────────────────────────────────────────────────────
    best_H       = H_base
    best_masks   = _score(H_base)
    best_count   = best_masks[3]

    # Minimum line samples per trial to reach 8 equations (with all keypoints).
    eqs_from_kp         = 2 * n_kp
    min_line_needed     = max(0, 8 - eqs_from_kp)
    total_line_pts      = n_h + n_v
    samples_per_trial   = max(min_line_needed, min(8, total_line_pts))

    rng = np.random.default_rng(42)

    for _ in range(ransac_iterations):
        if total_line_pts == 0:
            H_cand = _solve_subset(kp_img_n, kp_can_n, [], [], [], [])
        else:
            n_pick = min(samples_per_trial, total_line_pts)
            all_line_idx = [(0, i) for i in range(n_h)] + [(1, i) for i in range(n_v)]
            chosen = [all_line_idx[i] for i in rng.choice(len(all_line_idx), size=n_pick, replace=False)]

            hi_sub = [h_img_n[i] for t, i in chosen if t == 0]
            hy_sub = [h_y_n[i]   for t, i in chosen if t == 0]
            vi_sub = [v_img_n[i] for t, i in chosen if t == 1]
            vx_sub = [v_x_n[i]   for t, i in chosen if t == 1]

            H_cand = _solve_subset(kp_img_n, kp_can_n, hi_sub, hy_sub, vi_sub, vx_sub)

        if H_cand is None:
            continue

        masks = _score(H_cand)
        if masks[3] > best_count:
            best_count = masks[3]
            best_H     = H_cand
            best_masks = masks

    # ── 7. Final refit on all inliers ────────────────────────────────────────
    kp_in, h_in, v_in, _ = best_masks

    kp_i_fin = kp_img_n[kp_in]   if n_kp > 0 else np.empty((0, 2))
    kp_c_fin = kp_can_n[kp_in]   if n_kp > 0 else np.empty((0, 2))
    hi_fin   = [h_img_n[i] for i in range(n_h) if h_in[i]]
    hy_fin   = [h_y_n[i]   for i in range(n_h) if h_in[i]]
    vi_fin   = [v_img_n[i] for i in range(n_v) if v_in[i]]
    vx_fin   = [v_x_n[i]   for i in range(n_v) if v_in[i]]

    n_fin_eq = 2 * int(kp_in.sum()) + len(hi_fin) + len(vi_fin)
    if n_fin_eq >= 4:
        H_final = _solve_subset(kp_i_fin, kp_c_fin, hi_fin, hy_fin, vi_fin, vx_fin)
        if H_final is None:
            H_final = best_H
    else:
        H_final = best_H

    # ── 8. Diagnostics ────────────────────────────────────────────────────────
    repr_errors_kp: List[float] = []
    for i, (u, v) in enumerate(keypoints_image):
        proj = H_final @ np.array([u, v, 1.0])
        if abs(proj[2]) > 1e-10:
            proj = proj / proj[2]
            err = float(np.sqrt((proj[0] - keypoints_canvas[i, 0]) ** 2 +
                                (proj[1] - keypoints_canvas[i, 1]) ** 2))
        else:
            err = float('inf')
        repr_errors_kp.append(round(err, 2))

    line_errs: Dict[str, List[float]] = defaultdict(list)
    for i, ((u, v), y_known, lid) in enumerate(zip(h_img_raw, h_y_raw, h_line_ids)):
        if not h_in[i]:
            continue
        proj = H_final @ np.array([u, v, 1.0])
        if abs(proj[2]) > 1e-10:
            proj = proj / proj[2]
            line_errs[lid].append(abs(float(proj[1]) - y_known))
    for i, ((u, v), x_known, lid) in enumerate(zip(v_img_raw, v_x_raw, v_line_ids)):
        if not v_in[i]:
            continue
        proj = H_final @ np.array([u, v, 1.0])
        if abs(proj[2]) > 1e-10:
            proj = proj / proj[2]
            line_errs[lid].append(abs(float(proj[0]) - x_known))

    repr_errors_lines = {lid: round(float(np.mean(errs)), 2) for lid, errs in line_errs.items()}

    info: dict = {
        'num_keypoints':         n_kp,
        'num_line_points':       n_h + n_v,
        'num_lines':             len(line_annotations),
        'num_inliers':           int(best_count),
        'converged':             True,
        'repr_errors_keypoints': repr_errors_kp,
        'repr_errors_lines':     repr_errors_lines,
        'line_warnings':         line_warnings,
    }
    return H_final, info

