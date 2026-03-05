"""
Per-frame homography propagation via inter-frame feature matching.

Key correction vs previous version:
  - Feature matching is done in FULL IMAGE SPACE (not a cropped ROI).
    Using a ROI introduced a coordinate-space mismatch when composing Hs.
    We instead use a MASK to restrict where ORB detects features, which
    keeps keypoint coordinates in full-image space.
  - Composition direction is corrected: H_inter maps frame_curr→frame_prev
    (i.e. "where did this pixel come from"), so the chain gives
    H_abs[f] = H_anchor @ H_{f→anchor} correctly.
"""

import cv2
import numpy as np
import logging
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)

DEFAULT_MAX_FEATURES  = 2000
DEFAULT_MAX_MATCHES   = 500
DEFAULT_RANSAC_THRESH = 3.0

# Fraction of image HEIGHT to use for feature detection mask.
# Keypoints below this line (crowd, ads) are excluded.
# Coordinates remain in full-image space.
PITCH_MASK_HEIGHT_FRAC = 0.65


def _extract_frame(cap: cv2.VideoCapture, frame_idx: int) -> Optional[np.ndarray]:
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    return frame if ret else None


def _make_pitch_mask(h: int, w: int, frac: float = PITCH_MASK_HEIGHT_FRAC) -> np.ndarray:
    """Binary mask: 255 in upper pitch region, 0 in crowd/sky below."""
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[:int(h * frac), :] = 255
    return mask


def _match_frames(
    img_src: np.ndarray,  # "from" frame  (the one we want to map FROM)
    img_dst: np.ndarray,  # "to" frame    (the one we want to map TO)
    max_features: int = DEFAULT_MAX_FEATURES,
    max_matches:  int = DEFAULT_MAX_MATCHES,
    ransac_thresh: float = DEFAULT_RANSAC_THRESH,
) -> Tuple[Optional[np.ndarray], float, int]:
    """
    Compute H that maps img_src pixel coordinates → img_dst pixel coordinates.

    Feature detection uses a pitch-region mask but keypoint coords are
    always in full-image space, so H can be composed with other full-image Hs.

    Returns: (H_src_to_dst, inlier_ratio, n_inliers)
    """
    h, w = img_src.shape[:2]
    mask = _make_pitch_mask(h, w)

    g_src = cv2.cvtColor(img_src, cv2.COLOR_BGR2GRAY)
    g_dst = cv2.cvtColor(img_dst, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(max_features)
    kp_src, des_src = orb.detectAndCompute(g_src, mask)
    kp_dst, des_dst = orb.detectAndCompute(g_dst, mask)

    if des_src is None or des_dst is None or len(kp_src) < 8 or len(kp_dst) < 8:
        return None, 0.0, 0

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = sorted(bf.match(des_src, des_dst), key=lambda m: m.distance)[:max_matches]

    if len(matches) < 8:
        return None, 0.0, 0

    # Full-image coordinates — no offset correction needed
    pts_src = np.float32([kp_src[m.queryIdx].pt for m in matches])
    pts_dst = np.float32([kp_dst[m.trainIdx].pt for m in matches])

    H, mask_h = cv2.findHomography(pts_src, pts_dst, cv2.RANSAC, ransac_thresh)
    if H is None or mask_h is None:
        return None, 0.0, 0

    n_inliers    = int(mask_h.sum())
    inlier_ratio = n_inliers / len(matches)
    return H, inlier_ratio, n_inliers


def build_per_frame_homographies(
    video_path: str,
    anchor_homographies: Dict[int, np.ndarray],
    start_frame: int = 0,
    end_frame:   Optional[int] = None,
    min_inlier_ratio: float = 0.4,
    verbose: bool = True,
) -> Dict[int, np.ndarray]:
    """
    Build a per-frame absolute homography (image → canvas) for every frame.

    For each segment between two anchor frames [a1, a2]:

      Forward pass from a1:
        H_abs[f] = H_anchor[a1] @ H_{a1→f}
        where H_{a1→f} = H_{f-1→f} @ H_{f-2→f-1} @ ... @ H_{a1→a1+1}
        i.e. the chain of "where did frame f's pixels come from in a1"

      This is computed by matching consecutive frames and accumulating.

    At each anchor frame the chain RESETS to the known anchor H,
    so drift is bounded to at most one anchor interval.
    """
    if not anchor_homographies:
        raise ValueError("anchor_homographies must not be empty")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if end_frame is None:
        end_frame = total_frames - 1

    anchors_in_range = sorted(
        a for a in anchor_homographies if start_frame <= a <= end_frame
    )
    if not anchors_in_range:
        raise ValueError("No anchor frames within the requested range")

    per_frame_H: Dict[int, np.ndarray] = {}

    # Place anchor frames immediately — these are ground truth
    for a in anchors_in_range:
        per_frame_H[a] = anchor_homographies[a].astype(np.float64).copy()

    # ── Helper: propagate forward from anchor a1 to frame target ─────────────
    def _propagate_forward(a1: int, target: int):
        """
        Fill per_frame_H for frames (a1, target] by chaining inter-frame Hs
        forward from anchor a1.

        H_abs[f] = H_anchor[a1] @ H_{a1→f}

        H_{a1→f} is built as:
          H_{a1→f} = H_{f-1→f} @ H_{f-2→f-1} @ ... @ H_{a1→a1+1}

        Each H_{n→n+1} maps frame n pixels → frame n+1 pixels (forward).
        So H_{a1→f} maps anchor-frame pixels → frame-f pixels.
        Then H_anchor[a1] @ inv(H_{a1→f}) maps frame-f pixels → canvas.

        Wait — let's be precise about directions.

        We need H_abs[f] such that:  canvas_pt = H_abs[f] @ image_pt_f

        H_anchor[a1]:  canvas_pt = H_anchor[a1] @ image_pt_a1
        H_{a1→f}:      image_pt_f ≈ H_{a1→f} @ image_pt_a1  (forward, maps a1→f)

        Therefore:   image_pt_a1 = inv(H_{a1→f}) @ image_pt_f
        And:         canvas_pt = H_anchor[a1] @ inv(H_{a1→f}) @ image_pt_f

        So:  H_abs[f] = H_anchor[a1] @ inv(H_{a1→f})
        """
        nonlocal per_frame_H

        prev_img = _extract_frame(cap, a1)
        if prev_img is None:
            logger.error(f"Cannot extract anchor frame {a1}")
            return

        # H_chain accumulates H_{a1→f}: maps anchor-frame coords to current-frame coords
        H_chain = np.eye(3, dtype=np.float64)
        H_a1 = anchor_homographies[a1].astype(np.float64)

        n_fallbacks = 0
        for f in range(a1 + 1, target + 1):
            curr_img = _extract_frame(cap, f)
            if curr_img is None:
                logger.warning(f"Frame {f}: cannot extract, freezing H")
                per_frame_H[f] = H_a1 @ np.linalg.inv(H_chain)
                n_fallbacks += 1
                continue

            # H_step: maps prev_frame → curr_frame (forward direction)
            H_step, inlier_ratio, n_inliers = _match_frames(prev_img, curr_img)

            if H_step is None or inlier_ratio < min_inlier_ratio:
                logger.warning(
                    f"Frame {f}: weak match (ratio={inlier_ratio:.2f}, "
                    f"n={n_inliers}) — freezing H"
                )
                per_frame_H[f] = H_a1 @ np.linalg.inv(H_chain)
                n_fallbacks += 1
            else:
                # H_chain = H_{a1→f} = H_step @ H_{a1→f-1}
                # (first apply old chain to get from a1 to f-1, then apply step to get to f)
                H_chain = H_step @ H_chain
                per_frame_H[f] = H_a1 @ np.linalg.inv(H_chain)

            prev_img = curr_img

        if verbose and n_fallbacks > 0:
            print(f"    ⚠ segment [{a1}→{target}]: {n_fallbacks} frozen frames")

    # ── Helper: propagate backward from anchor a2 to frame target ────────────
    def _propagate_backward(a2: int, target: int):
        """
        Fill per_frame_H for frames [target, a2) by chaining backward from anchor a2.

        H_abs[f] = H_anchor[a2] @ H_{f→a2}
        where H_{f→a2} is the chain of forward inter-frame Hs from f to a2.
        We build this by stepping backward: at each step we match curr→next
        (next is closer to a2) and accumulate.
        """
        nonlocal per_frame_H

        next_img = _extract_frame(cap, a2)
        if next_img is None:
            logger.error(f"Cannot extract anchor frame {a2}")
            return

        H_chain = np.eye(3, dtype=np.float64)  # H_{f→a2}: starts as identity at a2
        H_a2 = anchor_homographies[a2].astype(np.float64)

        n_fallbacks = 0
        for f in range(a2 - 1, target - 1, -1):
            curr_img = _extract_frame(cap, f)
            if curr_img is None:
                logger.warning(f"Frame {f}: cannot extract, freezing H")
                per_frame_H[f] = H_a2 @ H_chain
                n_fallbacks += 1
                continue

            # H_step: maps curr_frame → next_frame (one step toward a2)
            H_step, inlier_ratio, n_inliers = _match_frames(curr_img, next_img)

            if H_step is None or inlier_ratio < min_inlier_ratio:
                logger.warning(
                    f"Frame {f}: weak match (ratio={inlier_ratio:.2f}) — freezing H"
                )
                per_frame_H[f] = H_a2 @ H_chain
                n_fallbacks += 1
            else:
                # H_chain = H_{f→a2} = H_{f+1→a2} @ H_{f→f+1}
                H_chain = H_chain @ H_step
                per_frame_H[f] = H_a2 @ H_chain

            next_img = curr_img

        if verbose and n_fallbacks > 0:
            print(f"    ⚠ segment [{target}←{a2}]: {n_fallbacks} frozen frames")

    # ── Process segments ──────────────────────────────────────────────────────
    # Between anchor pairs: forward from left anchor
    for i in range(len(anchors_in_range) - 1):
        a1 = anchors_in_range[i]
        a2 = anchors_in_range[i + 1]
        if verbose:
            print(f"  Forward: anchor {a1} → anchor {a2}")
        _propagate_forward(a1, a2 - 1)  # stop one before a2; a2 itself is already placed

    # Before first anchor: backward from first anchor
    if start_frame < anchors_in_range[0]:
        a0 = anchors_in_range[0]
        if verbose:
            print(f"  Backward: frame {start_frame} ← anchor {a0}")
        _propagate_backward(a0, start_frame)

    # After last anchor: forward from last anchor
    if anchors_in_range[-1] < end_frame:
        a_last = anchors_in_range[-1]
        if verbose:
            print(f"  Forward: anchor {a_last} → frame {end_frame}")
        _propagate_forward(a_last, end_frame)

    cap.release()

    covered  = len(per_frame_H)
    expected = end_frame - start_frame + 1
    if verbose:
        print(f"\nDone. Covered {covered}/{expected} frames ({100*covered/expected:.1f}%)")

    return per_frame_H