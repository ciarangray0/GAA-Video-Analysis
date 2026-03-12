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

# v3: canvas-Y boundary separating near-side (reliable) from far-side (validated).
# 300 px ≈ 30 m from the near goal — covers endline, 13m-box, 20m features.
_NEAR_SIDE_Y_PX = 300.0

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
    from pipeline.line_intersections import compute_line_intersections
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

        # Normalise lines to LineAnnotation objects for intersection derivation.
        line_objects: List[LineAnnotation] = []
        for line in lines:
            if isinstance(line, LineAnnotation):
                line_objects.append(line)
            elif isinstance(line, dict):
                line_objects.append(LineAnnotation(**line))

        # Derive additional keypoints from line-pair intersections and merge
        # with any manually clicked ones (manual takes precedence on same pitch_id).
        if line_objects:
            derived = compute_line_intersections(line_objects)
            merged: dict = {p.pitch_id: p for p in derived}
            merged.update({p.pitch_id: p for p in keypoints})
            keypoints = list(merged.values())

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
        line_dicts = [
            {"line_id": l.line_id, "u1": l.u1, "v1": l.v1, "u2": l.u2, "v2": l.v2}
            for l in line_objects
        ]

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


# =============================================================================
# DLT-based homography with genuine line constraints
# =============================================================================

def compute_homographies_with_lines_v3(
    annotations: Dict[int, Dict],
    num_samples_per_line: int = 10,
    ransac_iterations: int = 2000,
    ransac_threshold: float = 10.0,
    keypoint_weight: float = 3.0,
) -> Tuple[Dict[int, np.ndarray], Dict[int, dict]]:
    """Compute per-anchor homographies using genuine DLT line constraints.

    Each line annotation provides one-dimensional DLT constraints (one row per
    sample point) rather than synthetic full-point correspondences.  This
    eliminates the circular dependency in ``compute_homographies_with_lines``
    where synthetic points were constructed by projecting through the very H
    being estimated.

    The function follows the same contract as ``compute_homographies_with_lines``
    so the two can be compared directly:
    - Accepts the same annotation dict format.
    - Derives additional keypoints from line intersections before solving.
    - Applies the same bad-anchor interpolation post-processing.

    Args:
        annotations:          Dict mapping frame_idx → annotation dict
                              ``{"keypoints": [...], "lines": [...]}``.
        num_samples_per_line: Points sampled per line annotation.
        ransac_iterations:    RANSAC trials in the DLT solver.
        ransac_threshold:     Inlier threshold in canvas pixels.
        keypoint_weight:      DLT row weight multiplier for full keypoints.

    Returns:
        ``(homographies, computation_info)`` — same structure as
        ``compute_homographies_with_lines``.
    """
    from pipeline.line_constraints import compute_homography_dlt_with_lines
    from pipeline.line_intersections import compute_line_intersections
    from pipeline.schemas import PitchPoint, LineAnnotation

    homographies: Dict[int, np.ndarray] = {}
    computation_info: Dict[int, dict] = {}

    for frame_idx, ann in annotations.items():
        if isinstance(ann, list):
            keypoints = ann
            lines = []
        else:
            keypoints = ann.get("keypoints", [])
            lines = ann.get("lines", [])

        # Normalise lines to LineAnnotation objects.
        line_objects: List[LineAnnotation] = []
        for line in lines:
            if isinstance(line, LineAnnotation):
                line_objects.append(line)
            elif isinstance(line, dict):
                line_objects.append(LineAnnotation(**line))

        # Derive intersection keypoints and merge with manual ones.
        if line_objects:
            derived = compute_line_intersections(line_objects)
            merged: dict = {p.pitch_id: p for p in derived}
            merged.update({p.pitch_id: p for p in keypoints})
            keypoints = list(merged.values())

        # Resolve canvas coordinates, dropping any unrecognised pitch_id.
        valid_keypoints = []
        for p in keypoints:
            try:
                resolve_pitch_coordinates(p.pitch_id)
                valid_keypoints.append(p)
            except ValueError:
                pass
        keypoints = valid_keypoints

        line_dicts = [
            {"line_id": l.line_id, "u1": l.u1, "v1": l.v1, "u2": l.u2, "v2": l.v2}
            for l in line_objects
        ]

        n_eq = 2 * len(keypoints) + len(line_dicts) * num_samples_per_line
        if n_eq < 4:
            computation_info[frame_idx] = {
                'error': f'Too few constraints ({n_eq} equations)',
                'num_keypoints': len(keypoints),
                'num_lines': len(line_dicts),
                'converged': False,
            }
            continue

        pts_image = np.array([[p.x_img, p.y_img] for p in keypoints], dtype=np.float32) \
            if keypoints else np.empty((0, 2), dtype=np.float32)
        pts_canvas = np.array([
            _meters_to_canvas_pixels(*resolve_pitch_coordinates(p.pitch_id))
            for p in keypoints
        ], dtype=np.float32) if keypoints else np.empty((0, 2), dtype=np.float32)

        try:
            H, info = compute_homography_dlt_with_lines(
                pts_image, pts_canvas,
                line_dicts,
                num_samples_per_line=num_samples_per_line,
                ransac_iterations=ransac_iterations,
                ransac_threshold=ransac_threshold,
                keypoint_weight=keypoint_weight,
            )
        except ValueError as exc:
            computation_info[frame_idx] = {
                'error': str(exc),
                'num_keypoints': len(keypoints),
                'num_lines': len(line_dicts),
                'converged': False,
            }
            continue

        # Reprojection stats over all keypoints.
        if info['repr_errors_keypoints']:
            errs = info['repr_errors_keypoints']
            info['repr_mean'] = round(float(np.mean(errs)), 2)
            info['repr_max']  = round(float(np.max(errs)), 2)
        elif info['repr_errors_lines']:
            all_line_errs = list(info['repr_errors_lines'].values())
            info['repr_mean'] = round(float(np.mean(all_line_errs)), 2)
            info['repr_max']  = round(float(np.max(all_line_errs)), 2)

        info['quality'] = (
            'good' if info.get('repr_mean', float('inf')) <= _BAD_ANCHOR_THRESHOLD else 'bad'
        )

        homographies[frame_idx] = H
        computation_info[frame_idx] = info

    # ── Bad-anchor interpolation (same logic as compute_homographies_with_lines) ──
    sorted_frames = sorted(homographies.keys())
    good_frames = [
        f for f in sorted_frames
        if computation_info.get(f, {}).get('repr_mean', float('inf')) <= _BAD_ANCHOR_THRESHOLD
    ]
    if good_frames:
        for f in sorted_frames:
            if computation_info.get(f, {}).get('repr_mean', float('inf')) > _BAD_ANCHOR_THRESHOLD:
                left  = [g for g in good_frames if g <= f]
                right = [g for g in good_frames if g >= f]
                if left and right:
                    f_l, f_r = left[-1], right[0]
                    if f_l == f_r:
                        homographies[f] = homographies[f_l]
                    else:
                        alpha = (f - f_l) / (f_r - f_l)
                        homographies[f] = (1.0 - alpha) * homographies[f_l] + alpha * homographies[f_r]
                elif left:
                    homographies[f] = homographies[left[-1]]
                elif right:
                    homographies[f] = homographies[right[0]]

    return homographies, computation_info


# =============================================================================
# v3: Near-first homography with line-intersection keypoints
# =============================================================================

def _is_convex_quad(pts: np.ndarray) -> bool:
    """Return True if four ordered 2-D points form a convex quadrilateral.

    Convexity is confirmed when all consecutive edge cross-products share the
    same sign (all clockwise or all counter-clockwise turns).

    Args:
        pts: (4, 2) array of points in traversal order (e.g. TL→TR→BR→BL).

    Returns:
        True if convex, False if concave or degenerate.
    """
    signs = []
    for i in range(4):
        p0 = pts[i]
        p1 = pts[(i + 1) % 4]
        p2 = pts[(i + 2) % 4]
        d1 = p1 - p0
        d2 = p2 - p1
        signs.append(d1[0] * d2[1] - d1[1] * d2[0])
    return all(s > 0 for s in signs) or all(s < 0 for s in signs)


def compute_homography_v3(
    lines: List[LineAnnotation],
    manual_keypoints: List[PitchPoint],
    near_side_threshold_px: float = _NEAR_SIDE_Y_PX,
    far_side_max_error_px: float = 40.0,
) -> dict:
    """Compute a single-frame homography using the near-first v3 strategy.

    Steps
    -----
    1. Derive keypoints from line-pair intersections via
       ``compute_line_intersections``.
    2. Merge with *manual_keypoints* (manual entries override derived ones for
       the same ``pitch_id``).
    3. Resolve canvas coordinates; drop any point with an unknown ``pitch_id``.
    4. Split into *near-side* (canvas Y < *near_side_threshold_px*) and
       *far-side* (everything else).
    5. Compute an initial H from near-side points only (requires ≥ 4).
    6. Validate each far-side point; accept only those whose reprojection error
       under the near-side H is below *far_side_max_error_px*.
    7. Recompute a final H from near + accepted-far points.
    8. Convex-quadrilateral sanity check: project the four canvas corners
       through H⁻¹ and verify the result is a convex quad in image space.

    Args:
        lines: Line annotations for this frame (``LineAnnotation`` objects).
        manual_keypoints: Manually clicked keypoints, e.g. goal posts.
        near_side_threshold_px: Canvas-Y boundary for near/far split (px).
        far_side_max_error_px: Reprojection error ceiling for far-side points (px).

    Returns:
        dict with keys:

        - ``H``              (np.ndarray | None) — 3×3 homography, or None on failure.
        - ``error``          (str | None)        — Failure reason when H is None.
        - ``points``         (list[dict])        — Per-point diagnostics:
            ``pitch_id``, ``x_img``, ``y_img``, ``canvas_x``, ``canvas_y``,
            ``source`` (``"manual"``|``"derived"``),
            ``side``   (``"near"``|``"far"``),
            ``accepted`` (bool), ``error_px`` (float|None).
        - ``n_near``         (int)  — Near-side points used.
        - ``n_far_accepted`` (int)  — Accepted far-side points.
        - ``n_far_rejected`` (int)  — Rejected far-side points.
        - ``n_derived``      (int)  — Keypoints derived from line intersections.
        - ``n_manual``       (int)  — Manually provided keypoints.
        - ``sanity_check``   (bool) — Canvas-corners convex-quad check result.
    """
    from pipeline.line_intersections import compute_line_intersections

    # 1 + 2.  Derive intersection keypoints and merge with manual ones.
    derived = compute_line_intersections(lines)
    manual_by_id = {p.pitch_id: p for p in manual_keypoints}
    all_by_id = {p.pitch_id: p for p in derived}
    all_by_id.update(manual_by_id)  # manual takes precedence on duplicates

    # 3.  Resolve canvas coordinates; skip unknown pitch_ids.
    point_data: List[dict] = []
    for p in all_by_id.values():
        try:
            x_m, y_m = resolve_pitch_coordinates(p.pitch_id)
        except ValueError:
            continue
        cx, cy = _meters_to_canvas_pixels(x_m, y_m)
        point_data.append({
            'pitch_id': p.pitch_id,
            'x_img':    p.x_img,
            'y_img':    p.y_img,
            'canvas_x': cx,
            'canvas_y': cy,
            'source':   'manual' if p.pitch_id in manual_by_id else 'derived',
        })

    def _failure(error: str, n_near: int = 0, n_far: int = 0) -> dict:
        pts = [
            {**pd,
             'side':     'near' if pd['canvas_y'] < near_side_threshold_px else 'far',
             'accepted': False,
             'error_px': None}
            for pd in point_data
        ]
        return {
            'H': None, 'error': error, 'points': pts,
            'n_near': n_near, 'n_far_accepted': 0, 'n_far_rejected': n_far,
            'n_derived': len(derived), 'n_manual': len(manual_by_id),
            'sanity_check': False,
        }

    if len(point_data) < 4:
        return _failure(f'Not enough valid points ({len(point_data)} < 4)')

    # 4.  Split near / far.
    near = [pd for pd in point_data if pd['canvas_y'] < near_side_threshold_px]
    far  = [pd for pd in point_data if pd['canvas_y'] >= near_side_threshold_px]

    if len(near) < 4:
        return _failure(
            f'Not enough near-side points ({len(near)} < 4)',
            n_near=len(near), n_far=len(far),
        )

    # 5.  Initial H from near-side only.
    pts_img_near = np.array([[pd['x_img'], pd['y_img']] for pd in near], dtype=np.float32)
    pts_can_near = np.array([[pd['canvas_x'], pd['canvas_y']] for pd in near], dtype=np.float32)
    try:
        H_near = compute_homography(pts_img_near, pts_can_near)
    except ValueError as exc:
        return _failure(f'Near-side homography failed: {exc}', n_near=len(near), n_far=len(far))

    def _reproj_err(H: np.ndarray, x_img: float, y_img: float, cx: float, cy: float) -> float:
        proj = H @ np.array([x_img, y_img, 1.0], dtype=np.float64)
        proj /= proj[2]
        return float(np.sqrt((proj[0] - cx) ** 2 + (proj[1] - cy) ** 2))

    # 6.  Annotate near points; validate far points against H_near.
    for pd in near:
        pd['side']     = 'near'
        pd['accepted'] = True
        pd['error_px'] = round(_reproj_err(H_near, pd['x_img'], pd['y_img'], pd['canvas_x'], pd['canvas_y']), 2)

    far_accepted: List[dict] = []
    far_rejected: List[dict] = []
    for pd in far:
        pd['side'] = 'far'
        err = _reproj_err(H_near, pd['x_img'], pd['y_img'], pd['canvas_x'], pd['canvas_y'])
        pd['error_px'] = round(err, 2)
        if err < far_side_max_error_px:
            pd['accepted'] = True
            far_accepted.append(pd)
        else:
            pd['accepted'] = False
            far_rejected.append(pd)

    # 7.  Final H from near + accepted far.
    accepted = near + far_accepted
    pts_img_all = np.array([[pd['x_img'], pd['y_img']] for pd in accepted], dtype=np.float32)
    pts_can_all = np.array([[pd['canvas_x'], pd['canvas_y']] for pd in accepted], dtype=np.float32)
    try:
        H_final = compute_homography(pts_img_all, pts_can_all)
    except ValueError:
        H_final = H_near  # fall back to near-only H

    # Refresh reprojection errors under the final H.
    for pd in accepted:
        pd['error_px'] = round(_reproj_err(H_final, pd['x_img'], pd['y_img'], pd['canvas_x'], pd['canvas_y']), 2)

    # 8.  Convex-quad sanity check via H⁻¹.
    sanity_ok = False
    try:
        H_inv = np.linalg.inv(H_final)
        canvas_corners = np.array([
            [0.0,        0.0,        1.0],
            [float(OUT_W), 0.0,        1.0],
            [float(OUT_W), float(OUT_H), 1.0],
            [0.0,        float(OUT_H), 1.0],
        ])
        img_corners = []
        for c in canvas_corners:
            proj = H_inv @ c
            if abs(proj[2]) < 1e-10:
                break
            proj /= proj[2]
            img_corners.append(proj[:2])
        if len(img_corners) == 4:
            sanity_ok = _is_convex_quad(np.array(img_corners))
    except np.linalg.LinAlgError:
        sanity_ok = False

    return {
        'H':              H_final,
        'error':          None,
        'points':         accepted + far_rejected,
        'n_near':         len(near),
        'n_far_accepted': len(far_accepted),
        'n_far_rejected': len(far_rejected),
        'n_derived':      len(derived),
        'n_manual':       len(manual_by_id),
        'sanity_check':   sanity_ok,
    }

