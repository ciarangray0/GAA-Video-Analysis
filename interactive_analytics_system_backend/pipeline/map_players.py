"""Map player detections to pitch canvas coordinates.

Coordinate System:
==================
- Input: Image pixels (camera frame from video)
- Output: Pitch canvas pixels (e.g., 850 × 1400 fixed canvas)
"""
from typing import List, Dict, Optional, Set
import numpy as np

from pipeline.schemas import Detection, PlayerPitchPosition
from pipeline.homography import map_pixel_to_pitch


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
