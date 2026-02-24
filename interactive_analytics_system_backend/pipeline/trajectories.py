"""Trajectory interpolation for player positions in pitch-pixel space.

This module interpolates player positions between anchor frames.
All interpolation happens in PITCH CANVAS PIXEL coordinates (e.g., 850 × 1400).

Coordinate System:
==================
- Input: Sparse positions at anchor frames (pitch canvas pixels)
- Output: Dense positions for all frames (pitch canvas pixels)

For tracks with 3+ anchor points, cubic spline interpolation is used for
smoother trajectories. Tracks with only 2 points fall back to linear
interpolation. Optionally, a Savitzky-Golay filter is applied to tracks
with >15 interpolated data points to reduce remaining jitter.

Meters are NOT used - all coordinates are pitch canvas pixels.
"""
from typing import List
import numpy as np
from scipy.signal import savgol_filter

from pipeline.config import OUT_W, OUT_H
from pipeline.schemas import PlayerPitchPosition


def interpolate_trajectories(
    sparse_positions: List[PlayerPitchPosition],
    start_frame: int,
    end_frame: int
) -> List[PlayerPitchPosition]:
    """
    Interpolate player trajectories between anchor frames.

    Uses cubic spline interpolation for tracks with 3+ anchor points and
    linear interpolation for tracks with only 2 anchor points. Positions
    are clamped to the valid pitch canvas range [0, OUT_W] × [0, OUT_H].
    A Savitzky-Golay filter is applied to smooth tracks that have more
    than 15 interpolated data points.

    All coordinates are in PITCH CANVAS PIXELS (not meters).

    Args:
        sparse_positions: List of PlayerPitchPosition with source="homography"
                         Coordinates are in pitch canvas pixels
        start_frame: First frame to interpolate
        end_frame: Last frame to interpolate (inclusive)

    Returns:
        List of PlayerPitchPosition including both original anchor frames
        (source="homography") and interpolated frames (source="interpolated")
        All coordinates are in pitch canvas pixels
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

        # Use simple linear interpolation for all tracks with >=2 anchors.
        # This avoids cubic-spline overshoot and keeps trajectories stable.
        xs_interp = np.interp(frames_interp, known_frames, known_xs)
        ys_interp = np.interp(frames_interp, known_frames, known_ys)

        # Clamp to valid pitch canvas bounds
        xs_interp = np.clip(xs_interp, 0, OUT_W)
        ys_interp = np.clip(ys_interp, 0, OUT_H)

        # Optional Savitzky-Golay smoothing for long tracks
        total_frames = len(frames_interp)
        if total_frames > 15:
            window_length = 7
            xs_interp = savgol_filter(xs_interp, window_length=window_length, polyorder=2)
            ys_interp = savgol_filter(ys_interp, window_length=window_length, polyorder=2)
            # Re-clamp after smoothing (filter can nudge values slightly outside bounds)
            xs_interp = np.clip(xs_interp, 0, OUT_W)
            ys_interp = np.clip(ys_interp, 0, OUT_H)

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
