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

from typing import List, Tuple, Dict
import numpy as np
import cv2

from pipeline.config import OUT_W, OUT_H


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

# Pitch dimensions
PITCH_METERS_H = 145.0  # Total height including goal area


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

    Performs two checks:
    1. Both endpoints should project to similar Y values (line is horizontal)
    2. Projected Y should be reasonably close to expected Y

    Args:
        line_annotation: Dict with line_id, u1, v1, u2, v2
        H_initial: Initial homography estimate
        y_tolerance_pixels: Maximum allowed Y deviation

    Returns:
        Tuple of (is_valid, error_message)
        - is_valid: True if annotation passes validation
        - error_message: Empty string if valid, otherwise describes the issue
    """
    try:
        y_expected = get_line_y_canvas(line_annotation['line_id'])
    except ValueError as e:
        return False, str(e)

    # Project both endpoints through homography
    p1 = np.array([line_annotation['u1'], line_annotation['v1'], 1.0])
    p2 = np.array([line_annotation['u2'], line_annotation['v2'], 1.0])

    proj1 = H_initial @ p1
    proj2 = H_initial @ p2

    # Check for points at infinity
    if abs(proj1[2]) < 1e-10 or abs(proj2[2]) < 1e-10:
        return False, "Line endpoints project to infinity"

    # Normalize to get actual coordinates
    y1 = proj1[1] / proj1[2]
    y2 = proj2[1] / proj2[2]

    # Check 1: Both Y values should be similar (line is horizontal in world space)
    y_diff = abs(y1 - y2)
    if y_diff > y_tolerance_pixels:
        return False, (
            f"Line endpoints project to very different Y values: "
            f"{y1:.1f} vs {y2:.1f} (diff={y_diff:.1f}px). "
            f"This suggests the line annotation is incorrect."
        )

    # Check 2: Average projected Y should be close to expected Y
    y_avg = (y1 + y2) / 2
    y_error = abs(y_avg - y_expected)
    if y_error > y_tolerance_pixels * 1.5:
        return False, (
            f"Projected Y ({y_avg:.1f}px) is far from expected "
            f"({y_expected:.1f}px) for line '{line_annotation['line_id']}'. "
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
    ransac_threshold: float = 5.0
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
    }

    # Step 1: Compute initial homography from keypoints only
    H_current, mask = cv2.findHomography(
        pts_image_keypoints,
        pts_canvas_keypoints,
        cv2.RANSAC,
        ransac_threshold
    )

    if H_current is None:
        raise ValueError("Failed to compute initial homography from keypoints")

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

        # Weight keypoints by duplication
        # This gives keypoints higher influence than synthetic line points
        pts_image_weighted = np.vstack([pts_image_keypoints] * keypoint_weight)
        pts_canvas_weighted = np.vstack([pts_canvas_keypoints] * keypoint_weight)

        # Combine all points: weighted keypoints + synthetic line points
        all_pts_image = np.vstack([pts_image_weighted, pts_image_synthetic])
        all_pts_world = np.vstack([pts_canvas_weighted, pts_world_synthetic])

        # Re-compute homography with all points
        H_new, _ = cv2.findHomography(
            all_pts_image.astype(np.float32),
            all_pts_world.astype(np.float32),
            cv2.RANSAC,
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

