# Player Tracking Module

Covers `map_players.py` (filtering and mapping detections to pitch coordinates) and `trajectories.py` (interpolation and smoothing).

---

## `map_players.py`

### Purpose
Takes the raw YOLO+BotSort detections and a per-frame homography dict, and produces pitch-canvas pixel positions for every player in every frame.

---

### `filter_detections_for_mapping(detections) → List[Detection]`

Removes non-player detections before mapping.

**Rules:**
1. Any detection with `class_name == CLASS_BALL` is dropped outright. Ball detections are meaningless for player tracking.
2. Any `track_id` that has **at least one** detection classified as `CLASS_REFEREE` is flagged as a referee track. **All** detections for that `track_id` (including frames where it was misclassified as a player) are dropped.

The whole-track referee removal is important because BotSort maintains track identity across frames. A referee may be occasionally misclassified as a player in some frames; if the track ID is dropped globally, these misclassifications are cleaned up automatically.

Logs a summary of what was dropped.

---

### `map_players_to_pitch(detections, homographies, anchor_frame_indices=None) → List[PlayerPitchPosition]`

Maps each detection's **bottom-centre** bounding box point to pitch-canvas coordinates.

**Bottom-centre formula:**
```python
x_foot = (det.x1 + det.x2) / 2   # horizontal centre of bbox
y_foot = det.y2                    # bottom edge of bbox
x_pitch, y_pitch = map_pixel_to_pitch(x_foot, y_foot, H)
```

The bottom-centre approximates where the player's feet contact the ground. The feet are the correct contact point for projecting a standing player onto the pitch plane — using the head centre or bbox centre would introduce systematic error because the camera is not directly overhead.

**Source labels:**
- `"homography"` — the detection's frame matches an anchor frame in `anchor_frame_indices`.
- `"homography_interp"` — the frame uses a propagated (optical-flow-derived) H. Used by the interpolation step to select which positions to interpolate between.

If `anchor_frame_indices` is `None`, all positions get `"homography"`.

Detections whose frame has no homography entry are silently skipped.

---

## `trajectories.py`

### Purpose
Converts the sparse, per-detection player positions into dense, smoothed trajectories for playback.

### Module-level constants

| Constant | Value | Meaning |
|----------|-------|---------|
| `_DEFAULT_MAX_VEL_PX` | `4.0` | Max displacement per frame (px) |
| `_DEFAULT_SG_LONG_WIN` | `15` | SG window for tracks >20 frames |
| `_DEFAULT_SG_MID_WIN` | `11` | SG window for tracks 10–20 frames |
| `_SG_LONG_TRACK_MIN` | `20` | Length threshold for "long" track category |
| `_SG_MID_TRACK_MIN` | `10` | Length threshold for "mid" track category |

The max-velocity default of 4 px/frame corresponds to 10 m/s at 10 px/m and 25 fps — approximately the maximum sprint speed in Gaelic football.

---

### `_sg_window(n_frames, long_win, mid_win) → Optional[int]`

Selects the Savitzky-Golay window for a track of `n_frames` length:

| Track length | Window |
|-------------|--------|
| > 20 frames | `min(long_win, n_frames)` |
| 10–20 frames | `min(mid_win, n_frames)` |
| < 10 frames | `None` (no smoothing) |

The returned window is always odd (the SG filter requirement) — if the computed value is even, it is decremented by 1. Short tracks (< 10 frames) are not smoothed because a short trajectory has few data points and SG filtering can introduce edge artefacts.

---

### `_apply_max_velocity(xs, ys, max_vel) → None` (in-place)

Clamps frame-to-frame displacement to at most `max_vel` pixels.

**Algorithm:**
```
for i in 1..len(xs)-1:
    dist = hypot(xs[i]-xs[i-1], ys[i]-ys[i-1])
    if dist > max_vel:
        scale = max_vel / dist
        xs[i] = xs[i-1] + (xs[i]-xs[i-1]) * scale
        ys[i] = ys[i-1] + (ys[i]-ys[i-1]) * scale
```

When a step exceeds `max_vel`, the position is moved along the **same direction** but capped at `max_vel`. All subsequent positions are then relative to this corrected point — meaning the shift is NOT propagated forward. This is intentional: a single detection outlier causes only a one-frame correction, not a sustained drift.

---

### `interpolate_trajectories(sparse_positions, start_frame, end_frame, ...) → List[PlayerPitchPosition]`

Full smoothing pipeline per track.

**Input:** sparse `PlayerPitchPosition` objects with `source in ("homography", "homography_interp")`.

**Steps per track:**

1. **Filter to range:** only positions within `[start_frame, end_frame]` are used.

2. **Skip single-detection tracks:** tracks with only 1 detection are returned as-is (nothing to interpolate between).

3. **Linear interpolation:**
   ```python
   frames_track = np.arange(track_start, track_end + 1)
   xs = np.interp(frames_track, known_frames, known_xs)
   ys = np.interp(frames_track, known_frames, known_ys)
   ```
   `np.interp` fills every frame between the first and last detection. No extrapolation — frames before the first detection or after the last are not generated.

4. **Canvas clip:** `xs = np.clip(xs, 0, OUT_W)`, `ys = np.clip(ys, 0, OUT_H)`.

5. **Savitzky-Golay smoothing** (if track is long enough):
   ```python
   xs = savgol_filter(xs, window_length=win, polyorder=2)
   ys = savgol_filter(ys, window_length=win, polyorder=2)
   xs = np.clip(xs, 0, OUT_W)
   ys = np.clip(ys, 0, OUT_H)
   ```
   SG smoothing is applied to the **full interpolated sequence** including originally-detected frames. This is important: earlier versions only smoothed interpolated frames, leaving the detected frames as raw values. This caused jitter at every anchor because the transition from smoothed interpolated values to raw detected values produced discontinuities.

6. **Max-velocity clamping** (if `max_vel_px > 0`):
   Applies `_apply_max_velocity` then clips again.

7. **Source label preservation:**
   Detected frames retain their original `source` label (`"homography"` or `"homography_interp"`). Filled frames get `source = "interpolated"`.

**Output:** all positions sorted by `(frame_idx, track_id)`.
