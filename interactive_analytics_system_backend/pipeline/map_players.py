"""Map player detections to pitch canvas coordinates.

Updated to support per-frame homographies from inter-frame propagation,
implementing the CV expert's suggestion to use every frame rather than
only anchor frames.

Coordinate System:
==================
- Input: Image pixels (camera frame from video)
- Output: Pitch canvas pixels (e.g., 850 × 1400 fixed canvas)
"""
from typing import List, Dict, Optional, Set
import numpy as np

from pipeline.schemas import Detection, PlayerPitchPosition
from pipeline.homography import map_pixel_to_pitch
from pipeline.config import OUT_W, OUT_H


def map_players_to_pitch(
    detections: List[Detection],
    homographies: Dict[int, np.ndarray],
    out_w: int = OUT_W,
    out_h: int = OUT_H,
    use_nearest_anchor: bool = False,
    anchor_frame_indices: Optional[Set[int]] = None,
) -> List[PlayerPitchPosition]:
    """Map player detections to pitch canvas coordinates via homography.

    use_nearest_anchor=False (default): expects a per-frame homography dict
    covering every frame. use_nearest_anchor=True falls back to nearest anchor
    H for frames not in the dict (legacy behaviour).

    Returns PlayerPitchPosition objects with source labels:
      "homography"        — anchor frame H
      "homography_interp" — propagated per-frame H
      "homography_anchor" — nearest anchor fallback (legacy)
    """
    if not homographies:
        return []

    all_frames_sorted = sorted(homographies.keys())
    positions = []

    for det in detections:
        H = homographies.get(det.frame_idx)
        source = "homography"

        if H is None:
            if not use_nearest_anchor:
                # Per-frame H dict should cover all frames — if missing, skip
                continue
            # Legacy fallback: nearest anchor
            nearest = min(all_frames_sorted, key=lambda a: abs(a - det.frame_idx))
            H = homographies[nearest]
            source = "homography_anchor"
        else:
            # If anchor_frame_indices is provided, use it to determine source.
            # Otherwise treat all frames as anchor (legacy behaviour).
            if anchor_frame_indices is not None:
                if det.frame_idx not in anchor_frame_indices:
                    source = "homography_interp"
            # If anchor_frame_indices is None, source stays "homography"

        x_center = (det.x1 + det.x2) / 2
        y_bottom = det.y2

        x_pitch, y_pitch = map_pixel_to_pitch(x_center, y_bottom, H)

        positions.append(PlayerPitchPosition(
            frame_idx=det.frame_idx,
            track_id=det.track_id,
            x_pitch=x_pitch,
            y_pitch=y_pitch,
            source=source,
        ))

    return positions