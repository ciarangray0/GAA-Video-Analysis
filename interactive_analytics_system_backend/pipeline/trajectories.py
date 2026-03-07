"""Trajectory interpolation for player positions in pitch-pixel space.

This module interpolates player positions between anchor frames.
All interpolation happens in PITCH CANVAS PIXEL coordinates (e.g., 850 × 1400).

Coordinate System:
==================
- Input: Dense positions from per-frame ORB-propagated homographies (pitch canvas pixels)
- Output: Smoothed dense positions for all frames (pitch canvas pixels)

A Savitzky-Golay filter is applied per-track over each track's own frame span
to remove high-frequency jitter introduced by ORB homography propagation.
Positions are clamped to the valid pitch canvas range [0, OUT_W] × [0, OUT_H].

Meters are NOT used - all coordinates are pitch canvas pixels.
"""
from typing import List
import numpy as np
from scipy.signal import savgol_filter

from pipeline.config import OUT_W, OUT_H
from pipeline.schemas import PlayerPitchPosition

# Minimum window for SavGol (must be odd and > polyorder).
# At 25 fps, 11 frames ≈ 0.44 s — wide enough to smooth ORB drift while
# keeping genuine rapid direction changes visible.
_SAVGOL_WINDOW = 11
_SAVGOL_POLYORDER = 2

def interpolate_trajectories(
    sparse_positions: List[PlayerPitchPosition],
    start_frame: int,
    end_frame: int
) -> List[PlayerPitchPosition]:
    """Smooth and interpolate player trajectories across all frames.

    When called with dense ORB-mapped positions (one per frame), this function
    acts primarily as a per-track smoother: it applies a Savitzky-Golay filter
    over each track's own frame span to remove jitter introduced by ORB
    homography propagation, then fills any remaining gaps with linear
    interpolation.

    When called with sparse anchor-only positions it fills gaps linearly before
    smoothing, as before.

    All coordinates are in PITCH CANVAS PIXELS (not meters).

    Args:
        sparse_positions: List of PlayerPitchPosition (source="homography" or
                         "homography_interp").  Coordinates in pitch canvas pixels.
        start_frame: First frame of the output range.
        end_frame:   Last frame of the output range (inclusive).

    Returns:
        List of PlayerPitchPosition covering [start_frame, end_frame] for every
        track present in the input.  source is preserved from the input for
        frames that had a measured position; frames that were gap-filled by
        linear interpolation use source="interpolated".
        All coordinates are in pitch canvas pixels.
    """
    filtered = [
        p for p in sparse_positions
        if start_frame <= p.frame_idx <= end_frame
    ]

    if not filtered:
        return []

    # Group by track_id
    by_track: dict = {}
    for pos in filtered:
        by_track.setdefault(pos.track_id, []).append(pos)

    all_positions: List[PlayerPitchPosition] = []
    frames_range = np.arange(start_frame, end_frame + 1)

    for track_id, positions in by_track.items():
        positions_sorted = sorted(positions, key=lambda p: p.frame_idx)

        if len(positions_sorted) < 2:
            all_positions.extend(positions_sorted)
            continue

        known_frames = np.array([p.frame_idx for p in positions_sorted])
        known_xs = np.array([p.x_pitch for p in positions_sorted])
        known_ys = np.array([p.y_pitch for p in positions_sorted])

        # Determine the contiguous frame range for this track
        track_start = int(known_frames[0])
        track_end = int(known_frames[-1])
        track_frames = np.arange(track_start, track_end + 1)

        # Build a source-label array parallel to track_frames so we can
        # preserve the original source tags after smoothing.
        frame_to_source = {p.frame_idx: p.source for p in positions_sorted}

        # Linear interpolation over the track's own span to fill any gaps
        xs_track = np.interp(track_frames, known_frames, known_xs)
        ys_track = np.interp(track_frames, known_frames, known_ys)

        # Clamp before smoothing
        xs_track = np.clip(xs_track, 0, OUT_W)
        ys_track = np.clip(ys_track, 0, OUT_H)

        # Apply SavGol over the track's own span.
        # Require at least window+1 points so the filter is meaningful; fall
        # back to no smoothing for very short tracks.
        n_track = len(track_frames)
        if n_track > _SAVGOL_WINDOW:
            xs_track = savgol_filter(xs_track, window_length=_SAVGOL_WINDOW, polyorder=_SAVGOL_POLYORDER)
            ys_track = savgol_filter(ys_track, window_length=_SAVGOL_WINDOW, polyorder=_SAVGOL_POLYORDER)
            # Re-clamp: filter can nudge values fractionally outside bounds
            xs_track = np.clip(xs_track, 0, OUT_W)
            ys_track = np.clip(ys_track, 0, OUT_H)

        # Build output only for frames within the requested range
        track_frame_to_xy = {
            int(f): (float(xs_track[i]), float(ys_track[i]))
            for i, f in enumerate(track_frames)
        }

        for f in frames_range:
            fi = int(f)
            if fi < track_start or fi > track_end:
                continue
            x, y = track_frame_to_xy[fi]
            source = frame_to_source.get(fi, "interpolated")
            all_positions.append(PlayerPitchPosition(
                frame_idx=fi,
                track_id=track_id,
                x_pitch=x,
                y_pitch=y,
                source=source,
            ))

    all_positions.sort(key=lambda p: (p.frame_idx, p.track_id))
    return all_positions
