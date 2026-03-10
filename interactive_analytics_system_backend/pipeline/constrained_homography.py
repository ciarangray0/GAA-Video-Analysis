"""
Per-frame H using ORB forward composition only.
Fallback used when PTZ interpolation is not available.
"""
import cv2
import numpy as np
from typing import Dict, Tuple


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