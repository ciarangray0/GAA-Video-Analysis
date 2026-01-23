"""Trajectory interpolation for player positions."""
from typing import List
import numpy as np

from pipeline.schemas import PlayerPitchPosition


def interpolate_trajectories(
    sparse_positions: List[PlayerPitchPosition],
    start_frame: int,
    end_frame: int
) -> List[PlayerPitchPosition]:
    """
    Interpolate player trajectories between anchor frames using linear interpolation.
    
    Args:
        sparse_positions: List of PlayerPitchPosition with source="homography"
        start_frame: First frame to interpolate
        end_frame: Last frame to interpolate (inclusive)
    
    Returns:
        List of PlayerPitchPosition including both original anchor frames
        (source="homography") and interpolated frames (source="interpolated")
    """
    # Filter positions in range
    filtered = [
        p for p in sparse_positions
        if start_frame <= p.frame_idx <= end_frame
    ]
    
    if not filtered:
        return []
    
    # Group by track_id
    by_track = {}
    for pos in filtered:
        if pos.track_id not in by_track:
            by_track[pos.track_id] = []
        by_track[pos.track_id].append(pos)
    
    # Generate interpolated trajectories
    all_positions = []
    frames_interp = np.arange(start_frame, end_frame + 1)
    
    for track_id, positions in by_track.items():
        # Sort by frame
        positions_sorted = sorted(positions, key=lambda p: p.frame_idx)
        
        if len(positions_sorted) < 2:
            # Not enough points to interpolate, keep original
            all_positions.extend(positions_sorted)
            continue
        
        # Extract known frames and coordinates
        known_frames = np.array([p.frame_idx for p in positions_sorted])
        known_xs = np.array([p.x_pitch for p in positions_sorted])
        known_ys = np.array([p.y_pitch for p in positions_sorted])
        
        # Interpolate x and y separately
        xs_interp = np.interp(frames_interp, known_frames, known_xs)
        ys_interp = np.interp(frames_interp, known_frames, known_ys)
        
        # Create positions for all frames
        known_frame_set = set(known_frames)
        
        for i, frame_idx in enumerate(frames_interp):
            if frame_idx in known_frame_set:
                # Keep original anchor frame position
                original = next(p for p in positions_sorted if p.frame_idx == frame_idx)
                all_positions.append(original)
            else:
                # Interpolated position
                all_positions.append(PlayerPitchPosition(
                    frame_idx=int(frame_idx),
                    track_id=track_id,
                    x_pitch=float(xs_interp[i]),
                    y_pitch=float(ys_interp[i]),
                    source="interpolated"
                ))
    
    # Sort by frame_idx, then track_id
    all_positions.sort(key=lambda p: (p.frame_idx, p.track_id))
    
    return all_positions
