"""Homography computation and pixel-to-pitch mapping.

Coordinate System:
    Image pixels (camera) → Homography H → Pitch canvas pixels

The pitch canvas is OUT_W × OUT_H pixels (850 × 1400).  Meters are never used
after the destination points are set up.
"""
import logging
import re
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

from pipeline.config import OUT_H, OUT_W
from pipeline.gaa_pitch_config import GAA_PITCH_VERTICES, GAA_PITCH_WIDTH, GAA_PITCH_LENGTH
from pipeline.schemas import LineAnnotation, PitchPoint

_REPROJECTION_OUTLIER_PX = 30.0   # threshold above which a keypoint is labelled "outlier"
_REPROJECTION_HIGH_PX    = 15.0   # threshold above which a keypoint is labelled "high" error


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _meters_to_canvas_pixels(x_m: float, y_m: float) -> Tuple[float, float]:
    """Convert pitch vertex coordinates (meters) to canvas pixels."""
    return x_m / GAA_PITCH_WIDTH * OUT_W, y_m / GAA_PITCH_LENGTH * OUT_H



def _compute_coverage_score(
    pts_image: np.ndarray,
    img_w: int,
    img_h: int,
    grid_cols: int = 3,
    grid_rows: int = 2,
) -> float:
    """Return fraction of grid cells containing ≥1 keypoint (0.0–1.0).

    Divides the frame into a grid_cols × grid_rows grid and counts how many
    cells are occupied.  Higher = better spatial spread of annotations.
    """
    if len(pts_image) == 0 or img_w == 0 or img_h == 0:
        return 0.0
    occupied = set()
    for x, y in pts_image:
        col = min(int(x / img_w * grid_cols), grid_cols - 1)
        row = min(int(y / img_h * grid_rows), grid_rows - 1)
        occupied.add((col, row))
    return round(len(occupied) / (grid_cols * grid_rows), 2)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _hartley_normalize(pts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Normalize points so centroid = origin and mean distance from origin = √2.

    This is the standard Hartley normalization required before building a DLT
    matrix. Without it, products of image coords (0–1440) × canvas coords
    (0–1400) reach ~2M, making the SVD numerically unstable.

    Args:
        pts: Nx2 array of 2D points.

    Returns:
        (pts_normalized, T) where T is the 3×3 normalization transform such
        that pts_normalized = (T @ homogeneous(pts).T).T[:, :2].
    """
    centroid = pts.mean(axis=0)
    shifted = pts - centroid
    mean_dist = np.sqrt((shifted ** 2).sum(axis=1)).mean()
    if mean_dist < 1e-8:
        return pts.copy(), np.eye(3, dtype=np.float64)
    scale = np.sqrt(2.0) / mean_dist
    T = np.array([
        [scale, 0.0,   -scale * centroid[0]],
        [0.0,   scale, -scale * centroid[1]],
        [0.0,   0.0,    1.0],
    ], dtype=np.float64)
    pts_h = np.column_stack([pts, np.ones(len(pts))])
    pts_n = (T @ pts_h.T).T[:, :2]
    return pts_n.astype(np.float64), T


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def resolve_pitch_coordinates(pitch_id: str) -> Tuple[float, float]:
    """Return (x_meters, y_meters) for a pitch_id.

    Accepts named vertices from GAA_PITCH_VERTICES or the encoded format
    ``line_<name>_x<X>_y<Y>``.
    """
    if pitch_id in GAA_PITCH_VERTICES:
        return GAA_PITCH_VERTICES[pitch_id]
    match = re.match(r'^line_.+_x([-\d.]+)_y([-\d.]+)$', pitch_id)
    if match:
        return float(match.group(1)), float(match.group(2))
    raise ValueError(
        f"Unrecognized pitch_id: '{pitch_id}'. "
        "Must be a known vertex name or follow the 'line_<name>_x<X>_y<Y>' format."
    )


def compute_homography(
    pts_image: np.ndarray,
    pts_pitch_canvas: np.ndarray,
) -> np.ndarray:
    """Compute a 3×3 homography matrix (image pixels → pitch canvas pixels)."""
    H, _ = cv2.findHomography(
        pts_image.astype(np.float32),
        pts_pitch_canvas.astype(np.float32),
        cv2.RANSAC,
        5.0,
    )
    if H is None:
        raise ValueError("Failed to compute homography")
    return H


def map_pixel_to_pitch(
    x_img: float,
    y_img: float,
    H: np.ndarray,
) -> Tuple[float, float]:
    """Map an image pixel → pitch canvas coordinates via homography."""
    p = np.array([x_img, y_img, 1.0], dtype=np.float32)
    pitch = H @ p
    pitch /= pitch[2]
    return float(pitch[0]), float(pitch[1])


def _compute_reprojection_errors(
    H: np.ndarray,
    pts_image: np.ndarray,
    pts_canvas: np.ndarray,
) -> np.ndarray:
    """Return per-point reprojection errors (canvas pixels)."""
    errors = []
    for img_pt, can_pt in zip(pts_image, pts_canvas):
        proj = H @ np.array([img_pt[0], img_pt[1], 1.0], dtype=np.float64)
        proj /= proj[2]
        errors.append(float(np.sqrt((proj[0] - can_pt[0]) ** 2 + (proj[1] - can_pt[1]) ** 2)))
    return np.array(errors, dtype=np.float64)


def _fill_info(
    computation_info: dict,
    frame_idx: int,
    H: np.ndarray,
    keypoints,
    pts_image: np.ndarray,
    pts_canvas: np.ndarray,
    valid_lines: int,
    n_line_pts: int,
    img_width: Optional[int],
    img_height: Optional[int],
) -> None:
    """Populate computation_info[frame_idx] with reprojection errors and quality."""
    kp_errors = _compute_reprojection_errors(H, pts_image, pts_canvas).tolist()
    kp_details = [
        {
            "pitch_id": kp.pitch_id,
            "error_px": round(err, 2),
            "verdict": "outlier" if err > _REPROJECTION_OUTLIER_PX else "high" if err > _REPROJECTION_HIGH_PX else "good",
        }
        for kp, err in zip(keypoints, kp_errors)
    ]

    mean_err = float(np.mean(kp_errors)) if kp_errors else 0.0
    n_outliers = sum(1 for e in kp_errors if e > _REPROJECTION_OUTLIER_PX)
    coverage = (
        _compute_coverage_score(pts_image, img_width, img_height)
        if img_width and img_height else None
    )

    computation_info[frame_idx] = {
        "num_keypoints": len(kp_details),
        "keypoints": kp_details,
        "repr_mean": round(mean_err, 2) if kp_errors else None,
        "repr_max": round(float(np.max(kp_errors)), 2) if kp_errors else None,
        "coverage": coverage,
        "valid_lines": valid_lines,
        "synthetic_points": n_line_pts,
        "quality": (
            "bad" if (n_outliers > 0 or mean_err > _REPROJECTION_OUTLIER_PX)
            else "warning" if (mean_err > _REPROJECTION_HIGH_PX or (coverage is not None and coverage < 0.5))
            else "good"
        ),
    }


def compute_homographies_with_lines_v3(
    annotations: Dict[int, Dict],
    num_samples_per_line: int = 10,
    ransac_iterations: int = 2000,
    ransac_threshold: float = 5.0,
    keypoint_weight: float = 20.0,
    img_width: Optional[int] = None,
    img_height: Optional[int] = None,
) -> Tuple[Dict[int, np.ndarray], Dict[int, dict]]:
    """Compute anchor homographies: keypoints define H, lines reinforce it.

    Algorithm:
      1. H₀ = findHomography(keypoints, RANSAC) — the primary robust solution.
      2. Build a weighted DLT system with Hartley-normalised coordinates:
           • Each keypoint contributes 2 rows (full X+Y) at weight keypoint_weight.
           • Each horizontal line sample contributes 1 row (Y-only) at weight 1.
           • Each vertical sideline sample contributes 1 row (X-only) at weight 1.
         With default keypoint_weight=20 and ~4 kp vs ~30 line pts the ratio is
         ~5:1, so keypoints dominate and lines can only correct underdetermined
         directions (e.g. X-skew far from the goal area).
      3. Solve via weighted SVD; denormalise.
      4. Sanity-check: if the result is degenerate or has worse reprojection than
         H₀, fall back to H₀.

    Hartley normalisation is mandatory here: without it, products of image coords
    (~1000s) × canvas coords (~1000s) reach ~10⁶ in the DLT matrix, making SVD
    numerically unstable and producing a catastrophically rotated output.
    """
    from pipeline.line_constraints import (
        GAA_PITCH_LINES,
        GAA_PITCH_SIDELINES,
        sample_points_on_line,
    )

    homographies: Dict[int, np.ndarray] = {}
    computation_info: Dict[int, dict] = {}

    for frame_idx, ann in annotations.items():
        keypoints = ann.get("keypoints", [])
        lines = ann.get("lines", [])

        if len(keypoints) < 4:
            logger.warning(
                f"Frame {frame_idx}: skipped — only {len(keypoints)} keypoint(s), need ≥ 4"
            )
            computation_info[frame_idx] = {
                "error": f"Too few keypoints ({len(keypoints)} < 4)",
                "quality": "bad",
            }
            continue

        pts_image = np.array([[p.x_img, p.y_img] for p in keypoints], dtype=np.float64)
        pts_canvas = np.array(
            [_meters_to_canvas_pixels(*resolve_pitch_coordinates(p.pitch_id)) for p in keypoints],
            dtype=np.float64,
        )

        # Step 1 — Primary H from keypoints only (RANSAC)
        H0, _ = cv2.findHomography(
            pts_image.astype(np.float32),
            pts_canvas.astype(np.float32),
            cv2.RANSAC,
            ransac_threshold,
            maxIters=ransac_iterations,
        )
        if H0 is None:
            computation_info[frame_idx] = {"error": "RANSAC failed on keypoints", "quality": "bad"}
            continue

        # Parse line annotations once
        line_dicts: List[dict] = []
        for line in lines:
            if isinstance(line, LineAnnotation):
                line_dicts.append({
                    "line_id": line.line_id,
                    "u1": line.u1, "v1": line.v1,
                    "u2": line.u2, "v2": line.v2,
                })
            elif isinstance(line, dict):
                line_dicts.append(line)

        # If no line annotations just return the keypoint H
        if not line_dicts:
            homographies[frame_idx] = H0.astype(np.float64)
            _fill_info(computation_info, frame_idx, H0, keypoints, pts_image, pts_canvas,
                       valid_lines=0, n_line_pts=0, img_width=img_width, img_height=img_height)
            continue

        # Step 2 — Hartley-normalised weighted DLT
        # Normalise keypoint coords (used to compute T_img / T_canvas)
        pts_image_n, T_img = _hartley_normalize(pts_image)
        pts_canvas_n, T_canvas = _hartley_normalize(pts_canvas)

        rows: List[List[float]] = []
        weights: List[float] = []

        # Keypoint rows — high weight, full 2D constraint
        w_kp = float(keypoint_weight)
        for (u, v), (x, y) in zip(pts_image_n, pts_canvas_n):
            rows.append([u, v, 1, 0, 0, 0, -x * u, -x * v, -x])
            rows.append([0, 0, 0, u, v, 1, -y * u, -y * v, -y])
            weights.extend([w_kp, w_kp])

        # Line rows — low weight, 1D constraint per sample
        # The fixed canvas coordinate must be normalised using T_canvas.
        # T_canvas maps: x_n = scale_c*(x - cx_c), y_n = scale_c*(y - cy_c)
        scale_c = T_canvas[0, 0]
        cx_c, cy_c = -T_canvas[0, 2] / scale_c, -T_canvas[1, 2] / scale_c

        n_line_pts = 0
        valid_lines = 0
        for line_dict in line_dicts:
            line_id = line_dict["line_id"]
            img_pts_raw = sample_points_on_line(
                line_dict["u1"], line_dict["v1"],
                line_dict["u2"], line_dict["v2"],
                num_samples_per_line,
            ).astype(np.float64)

            # Normalise image sample points
            img_pts_h = np.column_stack([img_pts_raw, np.ones(len(img_pts_raw))])
            img_pts_n = (T_img @ img_pts_h.T).T[:, :2]

            if line_id in GAA_PITCH_LINES:
                _, y_c_raw = _meters_to_canvas_pixels(0.0, GAA_PITCH_LINES[line_id])
                y_c = scale_c * (y_c_raw - cy_c)   # normalised canvas Y
                for u, v in img_pts_n:
                    rows.append([0, 0, 0, u, v, 1, -y_c * u, -y_c * v, -y_c])
                    weights.append(1.0)
                    n_line_pts += 1
                valid_lines += 1

            elif line_id in GAA_PITCH_SIDELINES:
                x_c_raw, _ = _meters_to_canvas_pixels(GAA_PITCH_SIDELINES[line_id], 0.0)
                x_c = scale_c * (x_c_raw - cx_c)   # normalised canvas X
                for u, v in img_pts_n:
                    rows.append([u, v, 1, 0, 0, 0, -x_c * u, -x_c * v, -x_c])
                    weights.append(1.0)
                    n_line_pts += 1
                valid_lines += 1

        # Step 3 — Weighted SVD solve
        A = np.array(rows, dtype=np.float64)
        w_vec = np.array(weights, dtype=np.float64)
        _, _, Vt = np.linalg.svd(A * w_vec[:, np.newaxis], full_matrices=False)
        H_norm = Vt[-1].reshape(3, 3)

        # Denormalise: H = T_canvas⁻¹ @ H_norm @ T_img
        H = np.linalg.inv(T_canvas) @ H_norm @ T_img
        if abs(H[2, 2]) > 1e-10:
            H /= H[2, 2]

        # Step 4 — Sanity check: fall back to H₀ if degenerate or worse
        def _repr_mean(mat):
            errs = []
            for ip, cp in zip(pts_image, pts_canvas):
                p = mat @ np.array([ip[0], ip[1], 1.0])
                p /= p[2]
                errs.append(float(np.sqrt((p[0] - cp[0])**2 + (p[1] - cp[1])**2)))
            return float(np.mean(errs))

        if np.any(np.isnan(H)) or np.linalg.cond(H) > 1e8 or _repr_mean(H) > _repr_mean(H0) * 2:
            H = H0.astype(np.float64)

        homographies[frame_idx] = H
        _fill_info(computation_info, frame_idx, H, keypoints, pts_image, pts_canvas,
                   valid_lines=valid_lines, n_line_pts=n_line_pts,
                   img_width=img_width, img_height=img_height)

    return homographies, computation_info


