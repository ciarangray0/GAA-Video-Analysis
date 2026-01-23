"""Map player detections to pitch coordinates using homographies."""
from typing import List, Dict
import numpy as np

from pipeline.schemas import Detection, PlayerPitchPosition
from pipeline.homography import map_pixel_to_distorted_pitch
from pipeline.config import OUT_W, OUT_H, K1


def map_players_to_pitch(
    detections: List[Detection],
    homographies: Dict[int, np.ndarray],
    out_w: int = None,
    out_h: int = None,
    k1: float = None
) -> List[PlayerPitchPosition]:
    """
    Map player detections to pitch coordinates using homographies.
    
    Uses bounding box bottom-center point for mapping.
    
    Args:
        detections: List of Detection objects
        homographies: Dict mapping frame_idx to 3x3 homography matrix
        out_w: Output canvas width (defaults to OUT_W)
        out_h: Output canvas height (defaults to OUT_H)
        k1: Radial distortion coefficient (defaults to K1)
    
    Returns:
        List of PlayerPitchPosition objects with source="homography"
    """
    if out_w is None:
        out_w = OUT_W
    if out_h is None:
        out_h = OUT_H
    if k1 is None:
        k1 = K1
    
    positions = []
    
    for det in detections:
        # Skip if no homography for this frame
        if det.frame_idx not in homographies:
            continue
        
        H = homographies[det.frame_idx]
        
        # Use bounding box bottom-center
        x_center = (det.x1 + det.x2) / 2
        y_bottom = det.y2
        
        # Map to pitch coordinates
        x_pitch, y_pitch = map_pixel_to_distorted_pitch(
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
