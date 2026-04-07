"""Map player detections to pitch canvas coordinates.

Coordinate System:
==================
- Input: Image pixels (camera frame from video)
- Output: Pitch canvas pixels (e.g., 850 × 1400 fixed canvas)
"""
import logging
from typing import Dict, List, Optional, Set

import numpy as np

from pipeline.homography import map_pixel_to_pitch
from pipeline.schemas import Detection, PlayerPitchPosition, CLASS_BALL, CLASS_REFEREE

logger = logging.getLogger(__name__)

# Tracks with fewer raw detections than this are treated as bogus (tracker glitches,
# brief misidentifications). At 25 fps, 25 detections ≈ 1 second of visibility.
MIN_TRACK_DETECTIONS = 25


def filter_detections_for_mapping(detections: List[Detection]) -> List[Detection]:
    """Drop ball, referee, and bogus short tracks before player mapping.

    Rules:
    - Any detection with class_name == CLASS_BALL is dropped outright.
    - Any track_id that has at least one detection classified as CLASS_REFEREE
      is considered a referee track; ALL detections for that track_id are dropped.
    - Any track_id with fewer than MIN_TRACK_DETECTIONS raw detections is dropped
      entirely (these are tracker glitches that appear briefly then vanish).

    Returns the filtered list (players only).
    """
    referee_track_ids: Set[int] = {
        det.track_id for det in detections if det.class_name == CLASS_REFEREE
    }

    # Count raw detections per track (excluding balls)
    track_counts: dict = {}
    for det in detections:
        if det.class_name != CLASS_BALL:
            track_counts[det.track_id] = track_counts.get(det.track_id, 0) + 1

    bogus_track_ids: Set[int] = {
        tid for tid, count in track_counts.items()
        if count < MIN_TRACK_DETECTIONS
    }

    filtered = [
        det for det in detections
        if det.class_name != CLASS_BALL
        and det.track_id not in referee_track_ids
        and det.track_id not in bogus_track_ids
    ]

    n_ball = sum(1 for d in detections if d.class_name == CLASS_BALL)
    n_ref_dets = sum(1 for d in detections if d.track_id in referee_track_ids)
    n_bogus_dets = sum(1 for d in detections if d.track_id in bogus_track_ids)
    logger.info(
        f"Detection filter: {len(detections)} total → "
        f"dropped {n_ball} ball, {n_ref_dets} referee dets "
        f"({len(referee_track_ids)} referee track IDs: {sorted(referee_track_ids)}), "
        f"{n_bogus_dets} bogus dets "
        f"({len(bogus_track_ids)} short track IDs: {sorted(bogus_track_ids)}), "
        f"{len(filtered)} remaining"
    )

    return filtered


def map_players_to_pitch(
    detections: List[Detection],
    homographies: Dict[int, np.ndarray],
    anchor_frame_indices: Optional[Set[int]] = None,
) -> List[PlayerPitchPosition]:
    """Map player detections to pitch canvas coordinates via per-frame homography.

    Expects a per-frame homography dict covering every frame; detections whose
    frame has no entry are skipped.

    Returns PlayerPitchPosition objects with source labels:
      "homography"        — anchor frame H
      "homography_interp" — propagated per-frame H
    """
    if not homographies:
        return []

    positions = []

    for det in detections:
        H = homographies.get(det.frame_idx)
        if H is None:
            continue

        source = "homography"
        if anchor_frame_indices is not None and det.frame_idx not in anchor_frame_indices:
            source = "homography_interp"

        x_pitch, y_pitch = map_pixel_to_pitch((det.x1 + det.x2) / 2, det.y2, H)

        positions.append(PlayerPitchPosition(
            frame_idx=det.frame_idx,
            track_id=det.track_id,
            x_pitch=x_pitch,
            y_pitch=y_pitch,
            source=source,
        ))

    return positions
