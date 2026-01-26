"""Map player detections to pitch canvas coordinates.

This module maps YOLO+ByteTrack bounding box detections from image pixels
to pitch canvas pixels. The output coordinates are the canonical representation
used for all visualization and analysis.

Coordinate System:
==================
- Input: Image pixels (camera frame from video)
- Output: Pitch canvas pixels (e.g., 850 × 1400 fixed canvas)

The mapping applies:
1. Homography transformation (image → pitch canvas)
2. Radial distortion (ALWAYS applied, matches notebook)

All interpolation and downstream processing uses pitch canvas pixels.
Meters are NOT used in this pipeline.
"""
from typing import List, Dict
import numpy as np

from pipeline.schemas import Detection, PlayerPitchPosition
from pipeline.homography import map_pixel_to_pitch
from pipeline.config import OUT_W, OUT_H, K1


def map_players_to_pitch(
    detections: List[Detection],
    homographies: Dict[int, np.ndarray],
    out_w: int = None,
    out_h: int = None,
    k1: float = None
) -> List[PlayerPitchPosition]:
    """
    Map player detections to pitch canvas coordinates.

    ONLY maps detections at frames where we have a homography (anchor frames).
    Uses bounding box bottom-center point for mapping (player's feet position).

    Output coordinates are in PITCH CANVAS PIXELS with radial distortion applied.
    This matches the notebook's visualization output exactly.

    Intermediate frames should be filled via interpolation after this step.
    Interpolation operates in pitch-pixel space.

    Args:
        detections: List of Detection objects from YOLO+ByteTrack
        homographies: Dict mapping frame_idx to 3x3 homography matrix
        out_w: Pitch canvas width in pixels (defaults to OUT_W)
        out_h: Pitch canvas height in pixels (defaults to OUT_H)
        k1: Radial distortion coefficient (defaults to K1)

    Returns:
        List of PlayerPitchPosition objects with source="homography"
        Coordinates (x_pitch, y_pitch) are in PITCH CANVAS PIXELS
    """
    if out_w is None:
        out_w = OUT_W
    if out_h is None:
        out_h = OUT_H
    if k1 is None:
        k1 = K1

    if not homographies:
        return []

    positions = []
    
    for det in detections:
        # ONLY map at frames where we have a homography (anchor frames)
        if det.frame_idx not in homographies:
            continue
        
        H = homographies[det.frame_idx]

        # Use bounding box bottom-center (player's feet position)
        x_center = (det.x1 + det.x2) / 2
        y_bottom = det.y2
        
        # Map to pitch canvas pixels WITH distortion (always applied)
        x_pitch, y_pitch = map_pixel_to_pitch(
            x_center,
            y_bottom,
            H,
            out_w,
            out_h,
            k1
        )
        
        positions.append(PlayerPitchPosition(
            frame_idx=det.frame_idx,
            track_id=det.track_id,
            x_pitch=x_pitch,
            y_pitch=y_pitch,
            source="homography"
        ))
    
    return positions
