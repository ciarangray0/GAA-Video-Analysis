"""Per-frame homography propagation via Lucas-Kanade optical flow.

build_optical_flow_per_frame_H — LK optical flow with drift correction + SG smoothing.
"""
import logging
import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_LK_WIN_SIZE      = (21, 21)
_LK_MAX_LEVEL     = 3
_LK_FB_THRESH     = 1.0     # forward-backward consistency threshold (px)
_LK_RANSAC_THRESH = 3.0


def _lk_inter_frame_H(
    g1: np.ndarray,
    g2: np.ndarray,
    mask: np.ndarray,
    max_corners: int,
    corner_quality: float,
    min_distance: float,
) -> Tuple[Optional[np.ndarray], int]:
    """Lucas-Kanade optical flow homography from g1 → g2.

    Uses forward-backward consistency to discard moving-player tracks.
    Returns (H_{g1→g2}, n_inliers).  H is None when matching fails.
    """
    pts1 = cv2.goodFeaturesToTrack(
        g1, maxCorners=max_corners, qualityLevel=corner_quality,
        minDistance=min_distance, mask=mask,
    )
    if pts1 is None or len(pts1) < 8:
        return None, 0

    lk_params = dict(
        winSize=_LK_WIN_SIZE, maxLevel=_LK_MAX_LEVEL,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )

    pts2, st_fwd, _ = cv2.calcOpticalFlowPyrLK(g1, g2, pts1, None, **lk_params)
    if pts2 is None:
        return None, 0

    pts1_back, st_bwd, _ = cv2.calcOpticalFlowPyrLK(g2, g1, pts2, None, **lk_params)
    if pts1_back is None:
        return None, 0

    fb_dist = np.linalg.norm(
        pts1.reshape(-1, 2) - pts1_back.reshape(-1, 2), axis=1
    )
    keep = (st_fwd.flatten() == 1) & (st_bwd.flatten() == 1) & (fb_dist < _LK_FB_THRESH)

    src = pts1.reshape(-1, 2)[keep]
    dst = pts2.reshape(-1, 2)[keep]

    if len(src) < 8:
        return None, 0

    H, mask_h = cv2.findHomography(src, dst, cv2.RANSAC, _LK_RANSAC_THRESH)
    if H is None or mask_h is None or int(mask_h.sum()) < 8:
        return None, 0

    return H, int(mask_h.sum())


def build_optical_flow_per_frame_H(
    video_path: str,
    anchor_homographies: Dict[int, np.ndarray],
    total_frames: int,
    max_corners: int = 500,
    corner_quality: float = 0.01,
    min_distance: float = 10.0,
    mask_top_fraction: float = 0.35,
) -> Tuple[Dict[int, np.ndarray], dict]:
    """Propagate anchor homographies to every frame using Lucas-Kanade optical flow.

    Improvements over ORB chaining:
    - LK optical flow + forward-backward consistency filters out moving players.
    - Correct chaining direction: H[t] = H[t-1] @ inv(H_{t-1→t}).
    - Bidirectional drift correction blended linearly between anchor pairs.
    - Savitzky-Golay smoothing per H-element within each inter-anchor segment.

    Args:
        video_path:          Path to the source video file.
        anchor_homographies: Trusted anchor Hs keyed by frame index.
        total_frames:        Total number of frames in the video.
        max_corners:         Max corners for goodFeaturesToTrack.
        corner_quality:      Quality level for goodFeaturesToTrack.
        min_distance:        Min pixel distance between detected corners.
        mask_top_fraction:   Fraction of frame height to exclude from the top
                             (sky / stands / trees).

    Returns:
        per_frame_H: {frame_idx: 3×3 H} for every frame 0..total_frames-1.
        info:        Diagnostic dict.
    """
    from scipy.signal import savgol_filter

    anchor_list = sorted(anchor_homographies.keys())
    if not anchor_list:
        return {}, {}

    per_frame_H: Dict[int, np.ndarray] = {}
    failed_frames: List[int] = []
    drift_at_anchors: Dict[int, float] = {}
    corners_per_frame: Dict[int, int] = {}
    unsmoothed_segments: List[Tuple[int, int]] = []

    # ------------------------------------------------------------------
    # Phase 1: Sequential pass — build optical flow H_{t→t+1} for all t
    # ------------------------------------------------------------------
    of_Hs: Dict[int, Optional[np.ndarray]] = {}  # of_Hs[t] = H_{t→t+1}

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    g_prev: Optional[np.ndarray] = None
    mask_cache: Optional[np.ndarray] = None

    for t in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            logger.warning(f"Optical flow: failed to read frame {t}")
            break

        g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if t % 50 == 0:
            logger.info(f"Optical flow pass 1: frame {t}/{total_frames}")

        if g_prev is not None:
            if mask_cache is None or mask_cache.shape != g.shape:
                h_img = g.shape[0]
                top_rows = int(h_img * mask_top_fraction)
                mask_cache = np.zeros(g.shape, dtype=np.uint8)
                mask_cache[top_rows:, :] = 255

            H_tf, n_inliers = _lk_inter_frame_H(
                g_prev, g, mask_cache,
                max_corners, corner_quality, min_distance,
            )
            of_Hs[t - 1] = H_tf
            corners_per_frame[t - 1] = n_inliers
            if H_tf is None:
                failed_frames.append(t - 1)

        g_prev = g

    cap.release()

    # ------------------------------------------------------------------
    # Phase 2: Chain and drift-correct per inter-anchor segment
    # ------------------------------------------------------------------

    # Frames before the first anchor: use first anchor H directly
    first_anchor = anchor_list[0]
    for f in range(first_anchor):
        per_frame_H[f] = anchor_homographies[first_anchor]

    for seg_i, A in enumerate(anchor_list):
        per_frame_H[A] = anchor_homographies[A]

        B = anchor_list[seg_i + 1] if seg_i + 1 < len(anchor_list) else None

        if B is None:
            # After the last anchor: assign its H to all remaining frames
            for f in range(A + 1, total_frames):
                per_frame_H[f] = anchor_homographies[A]
            break

        # --- Forward chain A+1 .. B ---
        H_chain: Dict[int, np.ndarray] = {A: anchor_homographies[A]}

        for t in range(A + 1, B + 1):
            H_tf = of_Hs.get(t - 1)  # H_{(t-1) → t}
            if H_tf is None:
                H_chain[t] = H_chain[t - 1].copy()
            else:
                try:
                    H_inv = np.linalg.inv(H_tf)
                    H_new = H_chain[t - 1] @ H_inv
                    if abs(H_new[2, 2]) > 1e-10:
                        H_new = H_new / H_new[2, 2]
                    H_chain[t] = H_new
                except np.linalg.LinAlgError:
                    H_chain[t] = H_chain[t - 1].copy()

        # --- Drift correction: linearly blend H_drift over the segment ---
        H_chain_B = H_chain[B]
        try:
            H_drift = anchor_homographies[B] @ np.linalg.inv(H_chain_B)
        except np.linalg.LinAlgError:
            H_drift = np.eye(3, dtype=np.float64)

        drift_at_anchors[B] = float(np.linalg.norm(H_drift - np.eye(3)))

        I3 = np.eye(3, dtype=np.float64)
        for t in range(A + 1, B):
            alpha = (t - A) / (B - A)
            H_corr = (1.0 - alpha) * I3 + alpha * H_drift
            H_corrected = H_corr @ H_chain[t]
            if abs(H_corrected[2, 2]) > 1e-10:
                H_corrected = H_corrected / H_corrected[2, 2]
            per_frame_H[t] = H_corrected

        # Pin anchor frames exactly
        per_frame_H[A] = anchor_homographies[A]
        per_frame_H[B] = anchor_homographies[B]

    # ------------------------------------------------------------------
    # Phase 3: Savitzky-Golay smoothing per segment (per H element)
    # ------------------------------------------------------------------
    sg_window_default = 21  # larger window smooths more aggressively across longer segments
    sg_order = 2

    for seg_i, A in enumerate(anchor_list):
        B = anchor_list[seg_i + 1] if seg_i + 1 < len(anchor_list) else None
        if B is None:
            break

        seg_frames = list(range(A, B + 1))
        n_seg = len(seg_frames)

        if n_seg < 5:
            unsmoothed_segments.append((A, B))
            continue

        # Largest odd window ≤ min(default, n_seg)
        eff_window = min(sg_window_default, n_seg)
        if eff_window % 2 == 0:
            eff_window -= 1
        if eff_window < 3:
            unsmoothed_segments.append((A, B))
            continue

        H_stack = np.array([per_frame_H[f] for f in seg_frames])  # (n, 3, 3)
        H_smoothed = H_stack.copy()

        for i in range(3):
            for j in range(3):
                H_smoothed[:, i, j] = savgol_filter(
                    H_stack[:, i, j], window_length=eff_window, polyorder=sg_order,
                )

        for k, f in enumerate(seg_frames):
            H_s = H_smoothed[k]
            if abs(H_s[2, 2]) > 1e-10:
                H_s = H_s / H_s[2, 2]
            per_frame_H[f] = H_s

        # Re-pin anchor frames after smoothing
        per_frame_H[A] = anchor_homographies[A]
        per_frame_H[B] = anchor_homographies[B]

    info = {
        'num_frames': total_frames,
        'failed_frames': failed_frames,
        'drift_at_anchors': drift_at_anchors,
        'corners_per_frame': corners_per_frame,
        'smoothing_window': sg_window_default,
        'unsmoothed_segments': unsmoothed_segments,
    }

    logger.info(
        f"Optical flow propagation complete: {total_frames} frames, "
        f"{len(failed_frames)} failed OF pairs, "
        f"{len(unsmoothed_segments)} unsmoothed segments"
    )

    return per_frame_H, info