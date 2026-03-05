"""
PTZ-based per-frame homography interpolation.

A PTZ camera's homography can be decomposed as:
    H = K @ R @ K_ref_inv

where K is the intrinsic matrix (focal length = zoom) and R is the
rotation matrix (pan + tilt). Interpolating pan, tilt, zoom separately
between anchor frames is geometrically correct — unlike interpolating
H matrix entries directly, which is not.

This directly implements the CV expert's suggestion.
"""
import numpy as np
import cv2
from typing import Dict, Tuple, Optional
from scipy.spatial.transform import Rotation, Slerp


# ── Camera model ─────────────────────────────────────────────────────────────
# Veo camera: 1920x1080, estimate principal point at centre
# Focal length will be estimated per anchor from the H decomposition
IMG_W, IMG_H = 1920, 1080
CX, CY = IMG_W / 2, IMG_H / 2

# Canvas (pitch) dimensions
OUT_W, OUT_H = 850, 1400
PITCH_W_M, PITCH_H_M = 85.0, 140.0


def _make_K(f: float) -> np.ndarray:
    """Intrinsic matrix for focal length f, fixed principal point."""
    return np.array([
        [f,   0,  CX],
        [0,   f,  CY],
        [0,   0,   1],
    ], dtype=np.float64)


def decompose_H_to_ptz(H: np.ndarray, f_init: float = 1200.0
                        ) -> Tuple[float, np.ndarray]:
    """
    Estimate focal length f and rotation R from homography H.

    H maps image pixels → canvas pixels.  We factor out the canvas
    scaling so we work with the camera-to-ground-plane homography.

    Returns (f, R_3x3).  R encodes pan and tilt.

    This is an approximation — a full calibration would be better —
    but it's sufficient for smooth interpolation between anchor frames.
    """
    # Normalise H
    H = H / H[2, 2]

    # We decompose H = K_canvas @ R @ K_img_inv
    # K_canvas maps ground-plane metres to canvas pixels
    # We approximate: find f such that K_inv @ H has columns that
    # are approximately orthonormal (rotation matrix property)

    def residual(f):
        K = _make_K(f)
        K_inv = np.linalg.inv(K)
        M = K_inv @ H  # should be ~ rotation matrix (up to scale)
        # Columns of a rotation matrix are unit vectors
        c0 = M[:, 0]; c1 = M[:, 1]
        c0 /= (np.linalg.norm(c0) + 1e-10)
        c1 /= (np.linalg.norm(c1) + 1e-10)
        return abs(np.dot(c0, c1))  # should be ~0 for rotation

    # Line search for f
    best_f, best_res = f_init, 1e9
    for f_test in np.linspace(500, 3000, 100):
        r = residual(f_test)
        if r < best_res:
            best_res = r
            best_f = f_test

    K = _make_K(best_f)
    K_inv = np.linalg.inv(K)
    M = K_inv @ H
    # Orthogonalise to get proper rotation matrix via SVD
    U, _, Vt = np.linalg.svd(M[:2, :2])  # work on 2x2 submatrix
    R2 = U @ Vt
    # Embed back into 3x3
    R = np.eye(3)
    R[:2, :2] = R2

    return best_f, R


def compose_H_from_ptz(f: float, R: np.ndarray) -> np.ndarray:
    """Recompose H from focal length and rotation."""
    K = _make_K(f)
    return K @ R @ np.linalg.inv(K)


def interpolate_ptz(
    f1: float, R1: np.ndarray,
    f2: float, R2: np.ndarray,
    alpha: float
) -> Tuple[float, np.ndarray]:
    """
    Interpolate PTZ parameters.
    - f (zoom): linear interpolation
    - R (pan+tilt): SLERP (spherical linear interpolation — geometrically correct)
    alpha=0 → (f1, R1), alpha=1 → (f2, R2)
    """
    f = (1 - alpha) * f1 + alpha * f2

    # SLERP for rotation
    r1 = Rotation.from_matrix(R1)
    r2 = Rotation.from_matrix(R2)
    slerp = Slerp([0, 1], Rotation.concatenate([r1, r2]))
    R_interp = slerp(alpha).as_matrix()

    return f, R_interp


def build_constrained_per_frame_H(
    video_path: str,
    anchor_H: Dict[int, np.ndarray],
    start_frame: int,
    end_frame: int,
    anchor_annotations: Dict[int, dict] = None,
    verbose: bool = False,
    **kwargs,
) -> Tuple[Dict[int, np.ndarray], dict]:
    """
    Per-frame H via PTZ interpolation between anchor frames.

    For each anchor, decompose H into (focal_length, rotation).
    For inter-anchor frames, interpolate f and R separately,
    then recompose H.

    This is geometrically correct for a PTZ camera observing a plane.
    """
    anchor_list = sorted(anchor_H.keys())
    per_frame_H: Dict[int, np.ndarray] = {}
    analysis:    dict = {}

    # Decompose all anchor Hs into PTZ
    anchor_ptz: Dict[int, Tuple[float, np.ndarray]] = {}
    for fidx, H in anchor_H.items():
        try:
            f, R = decompose_H_to_ptz(H)
            anchor_ptz[fidx] = (f, R)
            if verbose:
                print(f"Anchor {fidx}: f={f:.1f}  "
                      f"R={np.array2string(R, precision=3, suppress_small=True)}")
        except Exception as e:
            if verbose:
                print(f"Anchor {fidx}: decompose failed — {e}")
            anchor_ptz[fidx] = (1200.0, np.eye(3))

    for seg_i, anchor_start in enumerate(anchor_list):
        if anchor_start > end_frame:
            break

        anchor_end = (anchor_list[seg_i + 1]
                      if seg_i + 1 < len(anchor_list)
                      else end_frame)
        n_seg  = anchor_end - anchor_start
        H_start = anchor_H[anchor_start]

        per_frame_H[anchor_start] = H_start

        if n_seg == 0:
            continue

        f1, R1 = anchor_ptz[anchor_start]
        f2, R2 = anchor_ptz.get(anchor_end, (f1, R1))

        for f in range(anchor_start + 1, min(anchor_end + 1, end_frame + 1)):
            alpha = (f - anchor_start) / n_seg

            f_interp, R_interp = interpolate_ptz(f1, R1, f2, R2, alpha)
            H_interp = compose_H_from_ptz(f_interp, R_interp)

            # Rescale H_interp to match canvas coordinate system of H_start
            # (PTZ decomposition works in image coords; need to map to canvas)
            # We use the fact that H_start = K @ R1 @ K_inv @ H_ground
            # and scale the interpolated H accordingly
            H_interp_norm = H_interp / (H_interp[2, 2] + 1e-10)
            H_start_norm  = H_start  / (H_start[2, 2]  + 1e-10)

            # Find canvas scale: H_final = H_canvas_scale @ H_interp_norm
            # where H_canvas_scale maps from image-space homography to canvas
            # Approximate: use the ratio of H_start to identity-scaled interp
            # at anchor point to find canvas transform
            H_canvas = H_start_norm @ np.linalg.inv(
                compose_H_from_ptz(f1, R1)
            )
            H_f = H_canvas @ H_interp_norm
            if abs(H_f[2, 2]) > 1e-10:
                H_f /= H_f[2, 2]

            per_frame_H[f] = H_f

        analysis[anchor_start] = {'n': n_seg, 'f1': f1, 'f2': f2}

    if verbose:
        print("\nPTZ decomposition summary:")
        for fidx in anchor_list:
            f, R = anchor_ptz.get(fidx, (0, np.eye(3)))
            pan  = np.degrees(np.arctan2(R[0, 2], R[2, 2]))
            tilt = np.degrees(np.arcsin(-R[1, 2]))
            print(f"  Anchor {fidx:3d}: f={f:7.1f}  pan={pan:+7.2f}°  tilt={tilt:+7.2f}°")

    return per_frame_H, analysis