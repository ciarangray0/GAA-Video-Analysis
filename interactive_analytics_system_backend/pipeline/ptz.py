"""PTZ (Pan/Tilt/Zoom) camera model for incremental per-frame estimation.

Overview
--------
This module implements a PTZ camera model that replaces independent per-anchor-frame
homographies with a physically-grounded decomposition into camera orientation and
focal length parameters.  The core idea (suggested by the CV expert) is:

1. **Pan/Tilt** estimation via inter-frame homography decomposition.
   Consecutive frames share a large planar overlap (the pitch), so we can
   compute a reliable homography and decompose it into a rotation matrix whose
   Euler angles give pan and tilt.

2. **Zoom** estimation via two complementary methods:
   a. Scale factor of the inter-frame homography determinant.
   b. Radial optical flow divergence (zoom in → outward flow from FOE;
      zoom out → inward flow).

3. **Per-frame PTZ propagation**: starting from an anchor frame whose absolute
   PTZ values are fixed by the user-supplied keypoint homography, we chain
   incremental PTZ changes through every subsequent frame.

4. **Analytical homography from PTZ**: given (pan, tilt, zoom) we reconstruct a
   3×3 homography that maps camera pixels to the canonical pitch canvas without
   requiring any new keypoint annotations.

Coordinate conventions
----------------------
- Pan  (φ): positive = camera rotates right  (yaw about world Y-axis, radians)
- Tilt (θ): positive = camera tilts up        (pitch about camera X-axis, radians)
- Zoom (z): dimensionless ratio relative to the anchor frame (1.0 = no change)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class PTZState:
    """Pan/tilt/zoom parameters for a single video frame.

    Attributes
    ----------
    frame_idx : int
        Zero-based index of the video frame.
    pan : float
        Estimated horizontal camera rotation in radians relative to the
        anchor frame.  Positive means the camera has panned to the right.
    tilt : float
        Estimated vertical camera rotation in radians relative to the
        anchor frame.  Positive means the camera has tilted upward.
    zoom : float
        Zoom scale factor relative to the anchor frame.  Values > 1 indicate
        a tighter (zoomed-in) field of view; values < 1 indicate a wider view.
    source : str
        How this state was estimated: ``"anchor"``, ``"homography_decomp"``,
        or ``"optical_flow"``.
    """

    frame_idx: int
    pan: float = 0.0
    tilt: float = 0.0
    zoom: float = 1.0
    source: str = "anchor"


# ---------------------------------------------------------------------------
# Homography decomposition into PTZ
# ---------------------------------------------------------------------------

def decompose_homography_ptz(
    H: np.ndarray,
    focal_length: float,
    cx: float,
    cy: float,
) -> Tuple[float, float, float]:
    """Decompose a homography into approximate pan, tilt, and zoom.

    For a PTZ camera undergoing pure rotation (no translation, planar scene),
    the inter-frame homography satisfies::

        H = K · R_rel · K⁻¹

    where K = [[f, 0, cx], [0, f, cy], [0, 0, 1]] is the camera intrinsic
    matrix and R_rel is the relative rotation.  Solving for R_rel and
    extracting Euler angles yields pan (yaw) and tilt (pitch).  The zoom
    ratio is estimated from the scale factor encoded in H.

    Parameters
    ----------
    H : np.ndarray
        3×3 inter-frame homography matrix (frame_a → frame_b).
    focal_length : float
        Approximate focal length of the camera in pixels.
    cx, cy : float
        Principal point of the camera (image centre is a safe default).

    Returns
    -------
    pan : float
        Horizontal rotation in radians.
    tilt : float
        Vertical rotation in radians.
    zoom : float
        Scale factor (> 1 = zoomed in, < 1 = zoomed out).
    """
    K = np.array([
        [focal_length, 0.0, cx],
        [0.0, focal_length, cy],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)

    K_inv = np.linalg.inv(K)
    H_norm = K_inv @ H.astype(np.float64) @ K

    # Zoom from the Frobenius-norm scale of the normalised homography.
    # For a pure-rotation homography H_norm ≈ R, so det(R) = 1.
    # A zoom factor s yields H_norm ≈ s·R, hence s ≈ det(H_norm)^(1/3).
    det = np.linalg.det(H_norm)
    zoom = float(abs(det) ** (1.0 / 3.0)) if det != 0 else 1.0

    # Recover approximate rotation by normalising by the scale factor.
    R_approx = H_norm / (zoom if zoom != 0 else 1.0)

    # Clamp to valid rotation range to guard against numerical drift.
    R_approx = _project_to_rotation(R_approx)

    # Extract pan (yaw about Y) and tilt (pitch about X) from R = Ry(φ)·Rx(θ).
    # Using the small-angle convention often used in PTZ control:
    #   R ≈ [[cos φ,  sin φ·sin θ,  sin φ·cos θ],
    #         [0,     cos θ,        -sin θ       ],
    #         [-sin φ, cos φ·sin θ,  cos φ·cos θ]]
    tilt = float(np.arcsin(-np.clip(R_approx[1, 2], -1.0, 1.0)))
    pan = float(np.arctan2(R_approx[0, 2], R_approx[2, 2]))

    return pan, tilt, zoom


def _project_to_rotation(M: np.ndarray) -> np.ndarray:
    """Project matrix *M* onto SO(3) via SVD (nearest rotation matrix)."""
    U, _, Vt = np.linalg.svd(M)
    R = U @ Vt
    # Ensure proper rotation (det = +1, not −1)
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    return R


# ---------------------------------------------------------------------------
# Zoom estimation via radial optical flow
# ---------------------------------------------------------------------------

def estimate_zoom_from_optical_flow(
    frame_a: np.ndarray,
    frame_b: np.ndarray,
    principal_point: Optional[Tuple[float, float]] = None,
) -> float:
    """Estimate the zoom factor between two consecutive frames via optical flow.

    A camera zoom produces a characteristic radially outward (zoom-in) or
    inward (zoom-out) pattern in the dense optical flow field.  We fit a
    scalar divergence model::

        flow ≈ α · (p − p₀)

    where p₀ is the focus of expansion (approximated as the principal point),
    p is each pixel location, and α is the zoom rate.  The zoom scale factor
    for the frame pair is then ``1 + α``.

    Parameters
    ----------
    frame_a, frame_b : np.ndarray
        Consecutive BGR or greyscale frames.  They must be the same size.
    principal_point : tuple of float, optional
        (cx, cy) of the camera.  Defaults to the image centre.

    Returns
    -------
    float
        Zoom scale factor.  > 1 means the camera zoomed in between the two
        frames; < 1 means it zoomed out; ≈ 1 means no zoom.
    """
    if frame_a.shape[:2] != frame_b.shape[:2]:
        raise ValueError(
            "Frames must have the same spatial dimensions for optical flow."
        )

    h, w = frame_a.shape[:2]
    cx, cy = principal_point if principal_point is not None else (w / 2.0, h / 2.0)

    gray_a = _to_gray(frame_a)
    gray_b = _to_gray(frame_b)

    # Compute dense Farneback optical flow.
    flow = cv2.calcOpticalFlowFarneback(
        gray_a, gray_b,
        None,
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0,
    )

    # Build displacement vectors relative to the principal point.
    xs, ys = np.meshgrid(np.arange(w, dtype=np.float32),
                         np.arange(h, dtype=np.float32))
    dx = xs - cx
    dy = ys - cy
    r2 = dx * dx + dy * dy

    # Avoid division by zero at the principal point.
    mask = r2 > 1.0

    # Radial component of the flow: flow · r_hat = (fu·dx + fv·dy) / r
    fu = flow[..., 0]
    fv = flow[..., 1]
    radial_flow = np.where(mask, (fu * dx + fv * dy) / np.sqrt(r2 + 1e-9), 0.0)

    # Least-squares estimate of the zoom rate α:
    # radial_flow ≈ α · sqrt(r2)  =>  α = mean(radial_flow / sqrt(r2))
    sqrt_r2 = np.sqrt(r2 + 1e-9)
    alpha = float(np.mean(radial_flow[mask] / sqrt_r2[mask]))

    zoom = 1.0 + alpha
    return max(0.1, zoom)  # Clamp to a physically reasonable range


def _to_gray(frame: np.ndarray) -> np.ndarray:
    """Convert *frame* to uint8 greyscale if needed."""
    if frame.ndim == 3 and frame.shape[2] == 3:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    return frame


# ---------------------------------------------------------------------------
# Inter-frame homography estimation
# ---------------------------------------------------------------------------

def estimate_inter_frame_homography(
    frame_a: np.ndarray,
    frame_b: np.ndarray,
    max_features: int = 2000,
    ransac_threshold: float = 4.0,
) -> Optional[np.ndarray]:
    """Estimate homography between two consecutive frames using feature matching.

    Uses ORB feature detection and RANSAC-based homography estimation.  The
    resulting matrix H maps pixel coordinates in *frame_a* to pixel
    coordinates in *frame_b*.

    Parameters
    ----------
    frame_a, frame_b : np.ndarray
        Consecutive BGR or greyscale frames.
    max_features : int
        Maximum number of ORB keypoints to detect per frame.
    ransac_threshold : float
        RANSAC reprojection threshold in pixels.

    Returns
    -------
    np.ndarray or None
        3×3 homography matrix, or ``None`` if estimation failed (too few
        matched inliers).
    """
    gray_a = _to_gray(frame_a)
    gray_b = _to_gray(frame_b)

    orb = cv2.ORB_create(nfeatures=max_features)
    kp_a, des_a = orb.detectAndCompute(gray_a, None)
    kp_b, des_b = orb.detectAndCompute(gray_b, None)

    if des_a is None or des_b is None or len(kp_a) < 4 or len(kp_b) < 4:
        return None

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(des_a, des_b)

    if len(matches) < 4:
        return None

    # Sort by descriptor distance and keep the best matches.
    matches = sorted(matches, key=lambda m: m.distance)
    good = matches[: max(4, len(matches) // 2)]

    pts_a = np.float32([kp_a[m.queryIdx].pt for m in good])
    pts_b = np.float32([kp_b[m.trainIdx].pt for m in good])

    H, mask = cv2.findHomography(pts_a, pts_b, cv2.RANSAC, ransac_threshold)
    if H is None:
        return None

    inliers = int(mask.sum()) if mask is not None else 0
    if inliers < 4:
        return None

    return H


# ---------------------------------------------------------------------------
# PTZ propagation along the video
# ---------------------------------------------------------------------------

def propagate_ptz(
    anchor_frame: int,
    anchor_H_to_pitch: np.ndarray,
    inter_frame_homographies: Dict[int, np.ndarray],
    focal_length: float,
    image_width: int,
    image_height: int,
) -> Dict[int, PTZState]:
    """Build per-frame PTZ states by chaining inter-frame homographies.

    Starting from an anchor frame whose mapping to the pitch canvas is known
    (``anchor_H_to_pitch``), we walk forward (and backward) through the video
    using the provided inter-frame homographies to accumulate incremental
    pan/tilt/zoom changes.

    The anchor frame receives ``pan=0, tilt=0, zoom=1`` (the reference pose).
    Every other frame's PTZ is computed by composing the chain of inter-frame
    homographies from the anchor.

    Parameters
    ----------
    anchor_frame : int
        Index of the anchor frame (PTZ reference, pan=0/tilt=0/zoom=1).
    anchor_H_to_pitch : np.ndarray
        3×3 homography that maps anchor-frame pixels to pitch canvas pixels.
        This defines the absolute reference for all per-frame homographies.
    inter_frame_homographies : dict
        Mapping ``frame_idx → H_inter`` where H_inter maps pixels in frame
        ``frame_idx − 1`` to pixels in frame ``frame_idx``.
    focal_length : float
        Approximate camera focal length in pixels (used for decomposition).
    image_width, image_height : int
        Frame dimensions (pixels) – used to determine the principal point.

    Returns
    -------
    dict
        Mapping ``frame_idx → PTZState`` for every frame that can be reached
        from the anchor through the inter-frame chain.
    """
    cx = image_width / 2.0
    cy = image_height / 2.0

    ptz_states: Dict[int, PTZState] = {}
    ptz_states[anchor_frame] = PTZState(
        frame_idx=anchor_frame,
        pan=0.0,
        tilt=0.0,
        zoom=1.0,
        source="anchor",
    )

    # Collect all frame indices reachable from the inter-frame homographies.
    all_frames = sorted(set(inter_frame_homographies.keys()) | {anchor_frame})

    # Walk forward from anchor.
    frames_after = [f for f in all_frames if f > anchor_frame]
    H_cumulative = np.eye(3, dtype=np.float64)
    for frame_idx in frames_after:
        H_inter = inter_frame_homographies.get(frame_idx)
        if H_inter is None:
            break
        H_cumulative = H_inter.astype(np.float64) @ H_cumulative
        pan, tilt, zoom = decompose_homography_ptz(H_cumulative, focal_length, cx, cy)
        ptz_states[frame_idx] = PTZState(
            frame_idx=frame_idx,
            pan=pan,
            tilt=tilt,
            zoom=zoom,
            source="homography_decomp",
        )

    # Walk backward from anchor.
    frames_before = [f for f in reversed(all_frames) if f < anchor_frame]
    H_cumulative = np.eye(3, dtype=np.float64)
    for frame_idx in frames_before:
        H_inter = inter_frame_homographies.get(frame_idx + 1)
        if H_inter is None:
            break
        # Invert: frame_idx → anchor direction.
        try:
            H_inter_inv = np.linalg.inv(H_inter.astype(np.float64))
        except np.linalg.LinAlgError:
            break
        H_cumulative = H_inter_inv @ H_cumulative
        pan, tilt, zoom = decompose_homography_ptz(H_cumulative, focal_length, cx, cy)
        ptz_states[frame_idx] = PTZState(
            frame_idx=frame_idx,
            pan=pan,
            tilt=tilt,
            zoom=zoom,
            source="homography_decomp",
        )

    return ptz_states


# ---------------------------------------------------------------------------
# Reconstruct per-frame pitch homography from PTZ state
# ---------------------------------------------------------------------------

def ptz_to_pitch_homography(
    ptz: PTZState,
    anchor_H_to_pitch: np.ndarray,
    focal_length: float,
    cx: float,
    cy: float,
) -> np.ndarray:
    """Reconstruct the pitch-canvas homography for a frame from its PTZ state.

    Given the anchor frame's absolute homography H_anchor (image → pitch
    canvas) and the PTZ delta of a target frame relative to the anchor, we
    synthesise the target frame's homography analytically::

        H_target = H_anchor · K · R_rel⁻¹ · K⁻¹

    where R_rel encodes the pan/tilt change and the zoom changes K.

    Parameters
    ----------
    ptz : PTZState
        PTZ state for the target frame (relative to the anchor).
    anchor_H_to_pitch : np.ndarray
        3×3 homography of the anchor frame (image pixels → pitch canvas).
    focal_length : float
        Camera focal length at the anchor zoom level, in pixels.
    cx, cy : float
        Principal point (camera image centre).

    Returns
    -------
    np.ndarray
        3×3 homography mapping target-frame image pixels to the pitch canvas.
    """
    # Build intrinsic matrix at anchor zoom (zoom=1).
    f_anchor = focal_length
    K_anchor = np.array([
        [f_anchor, 0.0, cx],
        [0.0, f_anchor, cy],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)

    # Build intrinsic matrix at target zoom.
    f_target = focal_length * ptz.zoom
    K_target = np.array([
        [f_target, 0.0, cx],
        [0.0, f_target, cy],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)

    # Rotation matrix for pan (φ) and tilt (θ).
    R = _rotation_from_pan_tilt(ptz.pan, ptz.tilt)

    # The homography from target frame to anchor frame is:
    #   H_target_to_anchor = K_anchor · R · K_target⁻¹
    K_target_inv = np.linalg.inv(K_target)
    H_target_to_anchor = K_anchor @ R @ K_target_inv

    # Chain with the anchor's absolute homography.
    H_target_to_pitch = anchor_H_to_pitch.astype(np.float64) @ H_target_to_anchor

    return H_target_to_pitch


def _rotation_from_pan_tilt(pan: float, tilt: float) -> np.ndarray:
    """Build a 3×3 rotation matrix from pan (yaw) and tilt (pitch) angles.

    Uses the right-handed convention::

        R = Ry(pan) · Rx(tilt)
    """
    cp, sp = np.cos(pan), np.sin(pan)
    ct, st = np.cos(tilt), np.sin(tilt)

    Ry = np.array([
        [cp,  0.0, sp],
        [0.0, 1.0, 0.0],
        [-sp, 0.0, cp],
    ], dtype=np.float64)

    Rx = np.array([
        [1.0, 0.0,  0.0],
        [0.0,  ct, -st],
        [0.0,  st,  ct],
    ], dtype=np.float64)

    return Ry @ Rx


# ---------------------------------------------------------------------------
# End-to-end helper: build per-frame homographies from frames + one anchor
# ---------------------------------------------------------------------------

def build_per_frame_homographies(
    frames: List[np.ndarray],
    anchor_frame_idx: int,
    anchor_H_to_pitch: np.ndarray,
    focal_length: Optional[float] = None,
    use_optical_flow_zoom: bool = True,
) -> Tuple[Dict[int, np.ndarray], Dict[int, PTZState]]:
    """Build a per-frame pitch homography for every frame using the PTZ model.

    This is the main entry-point for the PTZ pipeline.  It:

    1. Estimates inter-frame homographies between consecutive frames using
       ORB + RANSAC.
    2. Optionally refines the zoom estimate for each pair using dense optical
       flow.
    3. Propagates PTZ states from the anchor frame.
    4. Reconstructs a pitch-canvas homography for every reachable frame.

    Parameters
    ----------
    frames : list of np.ndarray
        Sequential list of BGR frames (index 0 = first frame in the clip).
    anchor_frame_idx : int
        Which frame (0-based) is the anchor (its ``anchor_H_to_pitch`` is
        taken as ground truth).
    anchor_H_to_pitch : np.ndarray
        3×3 homography that maps anchor-frame pixels to pitch canvas pixels.
    focal_length : float, optional
        Camera focal length in pixels.  Defaults to ``max(width, height)``
        which is a reasonable first approximation.
    use_optical_flow_zoom : bool
        If ``True``, refine the zoom factor for each consecutive pair using
        dense optical flow in addition to the homography scale factor.

    Returns
    -------
    homographies : dict
        Mapping ``frame_idx → 3×3 homography`` (image pixels → pitch canvas).
    ptz_states : dict
        Mapping ``frame_idx → PTZState`` for diagnostic / export purposes.
    """
    if not frames:
        return {}, {}

    h, w = frames[0].shape[:2]
    cx, cy = w / 2.0, h / 2.0

    if focal_length is None:
        focal_length = float(max(w, h))

    # Step 1 – estimate inter-frame homographies (frame i-1 → frame i).
    # Convention: frames[j] has clip-relative index j (0-based).
    # H under key i maps frames[i-1] → frames[i].
    inter_frame_H_abs: Dict[int, np.ndarray] = {}
    for i in range(1, len(frames)):
        H_inter = estimate_inter_frame_homography(frames[i - 1], frames[i])
        if H_inter is not None:
            inter_frame_H_abs[i] = H_inter  # maps frames[i-1] → frames[i]

    # Step 2 – optionally refine zoom via optical flow.
    if use_optical_flow_zoom:
        for i in range(1, len(frames)):
            if i not in inter_frame_H_abs:
                continue
            zoom_of = estimate_zoom_from_optical_flow(
                frames[i - 1], frames[i], (cx, cy)
            )
            H = inter_frame_H_abs[i].astype(np.float64)
            # Re-scale H so that its determinant matches the optical-flow zoom.
            det_H = np.linalg.det(H)
            scale_H = abs(det_H) ** (1.0 / 3.0) if det_H != 0 else 1.0
            if scale_H > 1e-6:
                H_rescaled = H * (zoom_of / scale_H)
                inter_frame_H_abs[i] = H_rescaled

    # Step 3 – propagate PTZ from anchor.
    ptz_states = propagate_ptz(
        anchor_frame=anchor_frame_idx,
        anchor_H_to_pitch=anchor_H_to_pitch,
        inter_frame_homographies=inter_frame_H_abs,
        focal_length=focal_length,
        image_width=w,
        image_height=h,
    )

    # Step 4 – reconstruct pitch homography for each frame.
    homographies: Dict[int, np.ndarray] = {
        anchor_frame_idx: anchor_H_to_pitch.astype(np.float64)
    }
    for frame_idx, ptz in ptz_states.items():
        if frame_idx == anchor_frame_idx:
            continue
        H_frame = ptz_to_pitch_homography(
            ptz, anchor_H_to_pitch, focal_length, cx, cy
        )
        homographies[frame_idx] = H_frame

    return homographies, ptz_states
