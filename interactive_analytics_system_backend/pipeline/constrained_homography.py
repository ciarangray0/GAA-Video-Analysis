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
    **kwargs,
) -> Tuple[Dict[int, np.ndarray], dict]:
    anchor_list = sorted(anchor_H.keys())
    per_frame_H: Dict[int, np.ndarray] = {}
    analysis: dict = {}

    cap = cv2.VideoCapture(video_path)

    for seg_i, anchor_start in enumerate(anchor_list):
        if anchor_start > end_frame:
            break

        anchor_end = (anchor_list[seg_i + 1]
                      if seg_i + 1 < len(anchor_list)
                      else end_frame)
        H_start = anchor_H[anchor_start]
        n_seg   = anchor_end - anchor_start

        per_frame_H[anchor_start] = H_start

        if n_seg == 0:
            continue

        # Seek once to the anchor frame, then read sequentially through the segment
        cap.set(cv2.CAP_PROP_POS_FRAMES, anchor_start)
        ret, f_prev_raw = cap.read()
        f_prev   = f_prev_raw if ret else None
        H_accum  = np.eye(3, dtype=np.float64)
        fwd_ok   = 0
        fallback = 0

        for f in range(anchor_start + 1, min(anchor_end + 1, end_frame + 1)):
            ret, f_curr_raw = cap.read()
            f_curr = f_curr_raw if ret else None
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
    return per_frame_H, analysis