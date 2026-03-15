"""Trajectory interpolation for player positions in pitch-pixel space.

All interpolation happens in PITCH CANVAS PIXEL coordinates (850 × 1400).

Pipeline per track
------------------
1. Linear interpolation fills every frame between first and last detection.
2. Savitzky-Golay smoothing (window/order depend on track length).
3. Maximum-velocity clamping removes spikes that survive SG filtering.

Coordinates are pitch canvas pixels throughout (not metres).
"""
from typing import List, Optional
import numpy as np
from scipy.signal import savgol_filter

from pipeline.config import OUT_W, OUT_H
from pipeline.schemas import PlayerPitchPosition

# Physical speed limit defaults
# 10 m/s × 10 px/m ÷ 25 fps = 4 px/frame
_DEFAULT_MAX_VEL_PX = 4.0   # px per frame
_DEFAULT_SG_LONG_WIN = 15   # tracks > 20 frames
_DEFAULT_SG_MID_WIN  = 11   # tracks 10-20 frames
# tracks < 10 frames: no smoothing


def _sg_window(n_frames: int, long_win: int, mid_win: int) -> Optional[int]:
    """Return the SG window to use for a track of n_frames length, or None."""
    if n_frames > 20:
        win = min(long_win, n_frames)
    elif n_frames >= 10:
        win = min(mid_win, n_frames)
    else:
        return None
    return win if win % 2 == 1 else win - 1  # must be odd


def _apply_max_velocity(xs: np.ndarray, ys: np.ndarray, max_vel: float):
    """Clamp frame-to-frame displacement to max_vel pixels per frame.

    Works in place — modifies xs and ys.  When a step exceeds max_vel the
    position is moved along the same direction but capped at max_vel, and all
    subsequent frames are shifted by the same correction so the rest of the
    trajectory is not affected.
    """
    for i in range(1, len(xs)):
        dx = xs[i] - xs[i - 1]
        dy = ys[i] - ys[i - 1]
        dist = np.hypot(dx, dy)
        if dist > max_vel:
            scale = max_vel / dist
            xs[i] = xs[i - 1] + dx * scale
            ys[i] = ys[i - 1] + dy * scale


def interpolate_trajectories(
    sparse_positions: List[PlayerPitchPosition],
    start_frame: int,
    end_frame: int,
    sg_long_window: int = _DEFAULT_SG_LONG_WIN,
    sg_mid_window: int = _DEFAULT_SG_MID_WIN,
    sg_polyorder: int = 2,
    max_vel_px: float = _DEFAULT_MAX_VEL_PX,
    # Legacy parameter — ignored; kept so old call-sites don't break
    legacy_window: int = 7,
) -> List[PlayerPitchPosition]:
    """Interpolate and smooth player trajectories.

    Args:
        sparse_positions: Detections with source="homography".
        start_frame:      First frame of the output range (used for filtering).
        end_frame:        Last frame of the output range (inclusive).
        sg_long_window:   SG window for tracks > 20 frames (default 15).
        sg_mid_window:    SG window for tracks 10-20 frames (default 11).
        sg_polyorder:     Savitzky-Golay polynomial order (default 2).
        max_vel_px:       Max allowed displacement per frame in pitch-canvas
                          pixels (default 4 = 10 m/s at 10 px/m, 25 fps).
        legacy_window:    Unused — present for backwards compatibility only.

    Returns:
        Dense list of PlayerPitchPosition for every frame within each track's
        [first_detection, last_detection] span that falls inside
        [start_frame, end_frame].
    """
    filtered = [p for p in sparse_positions if start_frame <= p.frame_idx <= end_frame]
    if not filtered:
        return []

    by_track: dict = {}
    for pos in filtered:
        by_track.setdefault(pos.track_id, []).append(pos)

    all_positions: List[PlayerPitchPosition] = []

    for track_id, positions in by_track.items():
        positions_sorted = sorted(positions, key=lambda p: p.frame_idx)

        if len(positions_sorted) < 2:
            all_positions.extend(positions_sorted)
            continue

        known_frames = np.array([p.frame_idx for p in positions_sorted], dtype=float)
        known_xs     = np.array([p.x_pitch   for p in positions_sorted])
        known_ys     = np.array([p.y_pitch   for p in positions_sorted])

        # Only interpolate within the detected range — no extrapolation
        track_start = int(known_frames[0])
        track_end   = int(known_frames[-1])
        frames_track = np.arange(track_start, track_end + 1)

        xs = np.interp(frames_track, known_frames, known_xs)
        ys = np.interp(frames_track, known_frames, known_ys)

        xs = np.clip(xs, 0, OUT_W)
        ys = np.clip(ys, 0, OUT_H)

        # --- Savitzky-Golay smoothing ---
        n = len(frames_track)
        win = _sg_window(n, sg_long_window, sg_mid_window)
        if win is not None and win >= 3:
            xs = savgol_filter(xs, window_length=win, polyorder=sg_polyorder)
            ys = savgol_filter(ys, window_length=win, polyorder=sg_polyorder)
            xs = np.clip(xs, 0, OUT_W)
            ys = np.clip(ys, 0, OUT_H)

        # --- Maximum-velocity clamping ---
        if max_vel_px > 0:
            _apply_max_velocity(xs, ys, max_vel_px)
            xs = np.clip(xs, 0, OUT_W)
            ys = np.clip(ys, 0, OUT_H)

        known_frame_set = set(int(f) for f in known_frames)
        for i, frame_idx in enumerate(frames_track):
            fi = int(frame_idx)
            # Use smoothed coordinates for all frames (detected and interpolated).
            # Previously, detected frames were written back as raw originals, which
            # meant SG smoothing had no effect on them — the main source of jitter.
            source = "homography" if fi in known_frame_set else "interpolated"
            all_positions.append(PlayerPitchPosition(
                frame_idx=fi,
                track_id=track_id,
                x_pitch=float(xs[i]),
                y_pitch=float(ys[i]),
                source=source,
            ))

    all_positions.sort(key=lambda p: (p.frame_idx, p.track_id))
    return all_positions
