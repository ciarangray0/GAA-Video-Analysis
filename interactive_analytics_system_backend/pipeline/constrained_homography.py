"""
Per-frame homography propagation.

build_constrained_per_frame_H  — ORB-based (used by v2 endpoint, unchanged)
build_optical_flow_per_frame_H — LK optical flow with drift correction + SG
                                  smoothing (used by v3 endpoint)
"""
import logging
import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


def _orb_inter_frame_H(f1, f2, n_features=2000, ransac_thresh=4.0):
    orb = cv2.ORB_create(n_features)
    g1 = cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(f2, cv2.COLOR_BGR2GRAY)
    kp1, d1 = orb.detectAndCompute(g1, None)
    kp2, d2 = orb.detectAndCompute(g2, None)
    if d1 is None or d2 is None or len(kp1) < 8:
        return None
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = sorted(bf.match(d1, d2), key=lambda m: m.distance)[:300]
    if len(matches) < 8:
        return None
    src = np.float32([kp1[m.queryIdx].pt for m in matches])
    dst = np.float32([kp2[m.trainIdx].pt for m in matches])
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, ransac_thresh)
    return H if (H is not None and mask.sum() >= 8) else None


def build_constrained_per_frame_H(
    video_path: str,
    anchor_H: Dict[int, np.ndarray],
    start_frame: int,
    end_frame: int,
    anchor_annotations: Dict[int, dict] = None,
    verbose: bool = False,
    anchor_quality: Dict[int, float] = None,
    quality_threshold: float = 30.0,
    **kwargs,
) -> Tuple[Dict[int, np.ndarray], dict]:
    # Build the effective anchor set, filtering out bad-quality anchors when
    # quality information is available.  This prevents a grossly wrong anchor H
    # from being used as the starting point for ORB-propagated segments.
    if anchor_quality is not None:
        effective_anchor_H = {
            k: v for k, v in anchor_H.items()
            if anchor_quality.get(k, 0.0) <= quality_threshold
        }
        if not effective_anchor_H:
            effective_anchor_H = anchor_H  # Fall back if all anchors are flagged bad
    else:
        effective_anchor_H = anchor_H

    anchor_list = sorted(effective_anchor_H.keys())
    per_frame_H: Dict[int, np.ndarray] = {}
    analysis: dict = {}

    cap = cv2.VideoCapture(video_path)

    def read_frame(idx: int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, f = cap.read()
        return f if ret else None

    for seg_i, anchor_start in enumerate(anchor_list):
        if anchor_start > end_frame:
            break

        anchor_end = (anchor_list[seg_i + 1]
                      if seg_i + 1 < len(anchor_list)
                      else end_frame)
        H_start = effective_anchor_H[anchor_start]
        n_seg   = anchor_end - anchor_start

        per_frame_H[anchor_start] = H_start

        if n_seg == 0:
            continue

        f_prev   = read_frame(anchor_start)
        H_accum  = np.eye(3, dtype=np.float64)
        fwd_ok   = 0
        fallback = 0

        for f in range(anchor_start + 1, min(anchor_end + 1, end_frame + 1)):
            f_curr = read_frame(f)
            if f_curr is None:
                per_frame_H[f] = H_start
                fallback += 1
                continue

            H_if = _orb_inter_frame_H(f_prev, f_curr)
            if H_if is not None:
                H_accum = H_if @ H_accum
                fwd_ok += 1
            else:
                fallback += 1

            try:
                H_inv = np.linalg.inv(H_accum)
                H_f = H_start @ H_inv
                if abs(H_f[2, 2]) > 1e-10:
                    H_f /= H_f[2, 2]
                per_frame_H[f] = H_f
            except np.linalg.LinAlgError:
                per_frame_H[f] = H_start
                fallback += 1

            f_prev = f_curr

        analysis[anchor_start] = {
            'fwd_ok': fwd_ok, 'fallback': fallback, 'n': n_seg
        }
        if verbose:
            print(f"Segment {anchor_start}→{anchor_end}: "
                  f"ORB ok={fwd_ok}/{n_seg} fallback={fallback}")

    cap.release()

    # Fill in any frames that precede the first (good) anchor by assigning
    # that anchor's H directly.  This handles the case where anchor frame 0
    # was filtered out as bad-quality and the first good anchor is later.
    if anchor_list:
        H_first = effective_anchor_H[anchor_list[0]]
        for f in range(start_frame, anchor_list[0]):
            if f not in per_frame_H:
                per_frame_H[f] = H_first

    return per_frame_H, analysis


# ---------------------------------------------------------------------------
# Optical-flow based propagation (used by v3 endpoint)
# ---------------------------------------------------------------------------

def _lk_inter_frame_H(
    g1: np.ndarray,
    g2: np.ndarray,
    mask: np.ndarray,
    max_corners: int,
    corner_quality: float,
    min_distance: float,
    fb_threshold: float = 1.0,
    ransac_thresh: float = 3.0,
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
        winSize=(21, 21), maxLevel=3,
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
    keep = (st_fwd.flatten() == 1) & (st_bwd.flatten() == 1) & (fb_dist < fb_threshold)

    src = pts1.reshape(-1, 2)[keep]
    dst = pts2.reshape(-1, 2)[keep]

    if len(src) < 8:
        return None, 0

    H, mask_h = cv2.findHomography(src, dst, cv2.RANSAC, ransac_thresh)
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