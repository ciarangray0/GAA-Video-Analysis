"""Jersey-colour-based team classification.

Classifies player tracks as 'ellistown' or 'opposition' by detecting the
fraction of Ellistown yellow pixels in the jersey region (top half of the
bounding box).

Ellistown wear a distinctive orange-yellow jersey (H=14-28 in OpenCV HSV).
Grass sits at H=35-40, providing a clean gap.

All thresholds are tunable constants at the top of this file.
"""
import logging
from typing import Dict, List

import cv2
import numpy as np

from pipeline.schemas import Detection

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tunable HSV thresholds (OpenCV ranges: H 0-179, S 0-255, V 0-255)
# ---------------------------------------------------------------------------

# Ellistown yellow — pixel data across 2 games shows jersey H=16-26.
# Grass sits at H=35-40, so there is a clean gap.
YELLOW_HUE_MIN = 14
YELLOW_HUE_MAX = 28
YELLOW_SAT_MIN = 100

# Grey/low-saturation pixels are masked before any analysis (glare, shadows)
GREY_SAT_THRESHOLD = 30

# Only inspect the top fraction of the bounding box (jersey, not legs/grass)
JERSEY_CROP_FRACTION = 0.5

# A player is Ellistown if at least this fraction of the (masked) jersey crop
# pixels fall in the yellow range.
MIN_YELLOW_FRACTION = 0.15


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def extract_jersey_yellow(frame: np.ndarray, bbox: tuple) -> tuple:
    """Extract yellow pixel fraction from the jersey region of a bounding box.

    Crops the top JERSEY_CROP_FRACTION of the bbox, converts to HSV, masks
    out low-saturation pixels, then counts what fraction of remaining pixels
    fall in the Ellistown yellow range.

    Args:
        frame: BGR video frame.
        bbox:  (x1, y1, x2, y2) bounding box in pixel coordinates.

    Returns:
        Tuple of:
          - mean_hsv: np.ndarray [H, S, V] of masked pixels (for swatch display).
          - yellow_fraction: float fraction of masked pixels that are Ellistown yellow.
    """
    x1, y1, x2, y2 = (int(v) for v in bbox)

    fh, fw = frame.shape[:2]
    x1 = max(0, min(x1, fw - 1))
    x2 = max(x1 + 1, min(x2, fw))
    y1 = max(0, min(y1, fh - 1))
    y2 = max(y1 + 1, min(y2, fh))

    # Crop to jersey region (top half)
    crop_y2 = y1 + max(1, int((y2 - y1) * JERSEY_CROP_FRACTION))
    crop = frame[y1:crop_y2, x1:x2]
    if crop.size == 0:
        return np.zeros(3, dtype=np.float32), 0.0

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV).astype(np.float32)
    sat_mask = hsv[:, :, 1] >= GREY_SAT_THRESHOLD
    if not np.any(sat_mask):
        return np.zeros(3, dtype=np.float32), 0.0

    masked = hsv[sat_mask]
    n = len(masked)

    yellow_mask = (
        (masked[:, 0] >= YELLOW_HUE_MIN) &
        (masked[:, 0] <= YELLOW_HUE_MAX) &
        (masked[:, 1] >= YELLOW_SAT_MIN)
    )

    return masked.mean(axis=0), float(yellow_mask.sum()) / n


def classify_tracks(
    video_path: str,
    detections: List[Detection],
    sample_frames: int = 30,
) -> Dict[int, dict]:
    """Classify each player track as 'ellistown' or 'opposition'.

    Samples frames for each track (evenly spaced across the track's duration),
    then does a single sequential forward pass through the video — each unique
    frame is decoded exactly once regardless of how many tracks appear in it.

    Referees should be filtered out before calling this function.

    Args:
        video_path:    Path to the video file.
        detections:    List of Detection objects (player tracks only).
        sample_frames: Maximum number of frames to sample per track.

    Returns:
        Dict mapping track_id → {team, confidence, mean_hsv}
    """
    by_track: Dict[int, List[Detection]] = {}
    for det in detections:
        by_track.setdefault(det.track_id, []).append(det)

    # Build frame_idx -> [(track_id, bbox), ...] so each frame is read once.
    frame_to_samples: Dict[int, List[tuple]] = {}
    for track_id, dets in by_track.items():
        dets_sorted = sorted(dets, key=lambda d: d.frame_idx)
        step = max(1, len(dets_sorted) // sample_frames)
        sampled = dets_sorted[::step][:sample_frames]
        for det in sampled:
            frame_to_samples.setdefault(det.frame_idx, []).append(
                (track_id, (det.x1, det.y1, det.x2, det.y2))
            )

    # Accumulators keyed by track_id.
    track_hsv: Dict[int, list] = {tid: [] for tid in by_track}
    track_yellow: Dict[int, list] = {tid: [] for tid in by_track}

    # Single forward pass — seeks are strictly non-decreasing.
    cap = cv2.VideoCapture(video_path)
    for frame_idx in sorted(frame_to_samples):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue
        for track_id, bbox in frame_to_samples[frame_idx]:
            hsv, yf = extract_jersey_yellow(frame, bbox)
            if np.any(hsv):
                track_hsv[track_id].append(hsv)
                track_yellow[track_id].append(yf)
    cap.release()

    classifications: Dict[int, dict] = {}
    for track_id in by_track:
        hsv_samples      = track_hsv[track_id]
        yellow_fractions = track_yellow[track_id]

        if not hsv_samples:
            classifications[track_id] = {
                "team": "opposition", "confidence": 0.0,
                "mean_hsv": [0.0, 0.0, 0.0],
            }
            continue

        mean_hsv = np.mean(hsv_samples, axis=0)
        h, s, v  = float(mean_hsv[0]), float(mean_hsv[1]), float(mean_hsv[2])

        mean_yellow = float(np.mean(yellow_fractions))

        if mean_yellow >= MIN_YELLOW_FRACTION:
            team = "ellistown"
            confidence = min(1.0, mean_yellow / (MIN_YELLOW_FRACTION * 2))
        else:
            team = "opposition"
            confidence = 1.0 - min(1.0, mean_yellow / max(MIN_YELLOW_FRACTION, 1e-6))

        classifications[track_id] = {
            "team":       team,
            "confidence": round(float(confidence), 3),
            "mean_hsv":   [round(h, 1), round(s, 1), round(v, 1)],
        }
        logger.debug(
            f"Track {track_id}: {team}  conf={confidence:.2f}  yellow={mean_yellow:.2f}"
        )


    n_ell = sum(1 for v in classifications.values() if v["team"] == "ellistown")
    n_opp = sum(1 for v in classifications.values() if v["team"] == "opposition")
    logger.info(
        f"Team classification: {n_ell} ellistown, {n_opp} opposition "
        f"from {len(by_track)} tracks"
    )

    return classifications


def override_classification(
    classifications: Dict[int, dict],
    track_id: int,
    new_team: str,
) -> Dict[int, dict]:
    """Override the team assignment for a single track.

    Args:
        classifications: Existing classification dict (not mutated).
        track_id:        The track to override.
        new_team:        New team string — "ellistown", "opposition", "referee", or "ignore".

    Returns:
        Updated copy of the classifications dict.
    """
    updated = dict(classifications)
    if track_id in updated:
        updated[track_id] = {**updated[track_id], "team": new_team}
    else:
        updated[track_id] = {"team": new_team, "confidence": 1.0,
                             "mean_hsv": [0.0, 0.0, 0.0]}
    return updated
