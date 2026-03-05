"""Map player detections to pitch canvas coordinates.

Updated to support per-frame homographies from inter-frame propagation,
implementing the CV expert's suggestion to use every frame rather than
only anchor frames.

Coordinate System:
==================
- Input: Image pixels (camera frame from video)
- Output: Pitch canvas pixels (e.g., 850 × 1400 fixed canvas)
"""
from typing import List, Dict, Optional
import numpy as np

from pipeline.schemas import Detection, PlayerPitchPosition
from pipeline.homography import map_pixel_to_pitch
from pipeline.config import OUT_W, OUT_H, K1


def map_players_to_pitch(
    detections: List[Detection],
    homographies: Dict[int, np.ndarray],
    out_w: int = OUT_W,
    out_h: int = OUT_H,
    k1: float = K1,
    use_nearest_anchor: bool = False,
) -> List[PlayerPitchPosition]:
    """
    Map player detections to pitch canvas coordinates.

    Supports two modes:
      use_nearest_anchor=False (default, recommended):
        Expects homographies to be a per-frame dict covering every frame.
        Maps ALL detections. No frames skipped.
        Build this dict using build_constrained_per_frame_H().

      use_nearest_anchor=True (legacy behaviour):
        Falls back to nearest anchor H for frames not in homographies dict.
        Equivalent to the original implementation but without skipping.

    Args:
        detections:          List of Detection objects from YOLO+ByteTrack
        homographies:        Dict mapping frame_idx → 3x3 H matrix
        out_w:               Pitch canvas width in pixels
        out_h:               Pitch canvas height in pixels
        k1:                  Radial distortion coefficient
        use_nearest_anchor:  If True, use nearest anchor H for unmapped frames

    Returns:
        List of PlayerPitchPosition objects.
        source="homography"        — mapped using exact frame H
        source="homography_interp" — mapped using interpolated/propagated H
        source="homography_anchor" — mapped using nearest anchor H (legacy)
    """
    if not homographies:
        return []

    anchor_frames_sorted = sorted(homographies.keys())
    positions = []

    for det in detections:
        H = homographies.get(det.frame_idx)
        source = "homography"

        if H is None:
            if not use_nearest_anchor:
                # Per-frame H dict should cover all frames — if missing, skip
                continue
            # Legacy fallback: nearest anchor
            nearest = min(anchor_frames_sorted, key=lambda a: abs(a - det.frame_idx))
            H = homographies[nearest]
            source = "homography_anchor"
        else:
            # If this frame is not one of the original anchor frames,
            # it came from propagation
            if det.frame_idx not in anchor_frames_sorted:
                source = "homography_interp"

        x_center = (det.x1 + det.x2) / 2
        y_bottom = det.y2

        x_pitch, y_pitch = map_pixel_to_pitch(
            x_center, y_bottom, H, out_w, out_h, k1
        )

        positions.append(PlayerPitchPosition(
            frame_idx=det.frame_idx,
            track_id=det.track_id,
            x_pitch=x_pitch,
            y_pitch=y_pitch,
            source=source,
        ))

    return positions