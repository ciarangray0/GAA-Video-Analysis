# Player Tracking Module

Covers `map_players.py` (filtering raw detections and projecting players onto the pitch diagram) and `trajectories.py` (filling gaps and smoothing those positions for playback).

---

## The big picture — what problem are we solving?

YOLO+BotSort watches every video frame and outputs a list of bounding boxes. Each box says "there is something at these pixel coordinates in frame N, and I think it belongs to track ID 42". That is all it knows — pixel coordinates in the camera image.

We need pitch coordinates, not camera pixels. We also need a position for every frame, not just the frames where the detector fired. And we need to throw away anything that is not actually a player (the ball, referees).

These two files handle those three jobs in order:

1. `map_players.py` — filter junk, then project each detection onto the pitch canvas
2. `trajectories.py` — fill the gaps with interpolation, then smooth the resulting paths

---

## `map_players.py`

### Why feet, not the centre of the box?

Imagine you are looking at a person from a slightly raised angle — the camera is mounted high in a stadium, not directly overhead. The person's head is farther from the ground than their feet. If you project the centre of the bounding box onto the pitch plane, you are projecting a point that is roughly at chest height. Because the camera is angled, chest height and ground level are at different places in the image — the chest projects to a point several meters away from where the player actually is standing.

Projecting the bottom-centre of the bounding box (directly below the midpoint at the lowest pixel row) gives you approximately where the player's feet touch the grass. Feet are on the ground plane, and the homography was estimated for the ground plane, so this is the only point guaranteed to project correctly.

```
Bottom-centre formula:
    x_foot = (x1 + x2) / 2    ← midpoint of left and right edges of the box
    y_foot = y2                ← the very bottom row of the box
```

For a box from pixel (300, 100) to pixel (360, 220), the foot point is (330, 220).

---

### `filter_detections_for_mapping` — removing noise before projection

Before projecting anything, we clean the detection list.

**Rule 1 — Drop the ball.**
The ball is a detection with `class_name == CLASS_BALL`. We have no interest in mapping the ball to the pitch (it does not sit on the ground plane reliably), so every ball detection is removed outright.

**Rule 2 — Drop referee tracks entirely.**
This one is trickier. The tracker assigns each person a `track_id` that persists across many frames. A referee wearing a distinctive coloured jersey will usually get classified as a referee. But because detectors are imperfect, the same physical person might be labelled "player" in frame 45 and "referee" in frame 60.

If we only dropped frames labelled "referee", we would keep the frame-45 detection — which is still a referee. That ghost detection would appear as a rogue player on the pitch diagram.

The fix: if a track ID has **ever** been labelled "referee" in any frame, drop **all** detections for that track ID, including the ones labelled "player". The reasoning is that a real player's track will never touch the referee label, so any track that does must belong to a referee.

```
Pseudocode:
    referee_track_ids = { det.track_id for det in detections if det.class_name == "referee" }
    keep = [ det for det in detections
             if det.class_name != "ball"
             and det.track_id not in referee_track_ids ]
```

---

### `map_players_to_pitch` — the actual projection

Once the list is clean, each detection is projected through a homography matrix (a 3×3 transformation that maps camera pixels to pitch canvas pixels).

The function looks up which homography to use for a given frame index. If the frame is an anchor frame (a frame the user annotated), it uses that frame's own homography directly. For every other frame it uses a propagated homography — one that was estimated by optical flow from the nearest anchor. The `source` label records which case applied:

- `"homography"` — directly computed from user annotations on this exact frame
- `"homography_interp"` — propagated from a nearby anchor via optical flow

This label matters later. The trajectory builder uses it to know which positions were "solid ground truth" and which were estimated.

If a frame has no homography at all (this can happen if propagation failed for that frame), the detection is silently skipped. No crash, just no output.

---

## `trajectories.py`

### The problem: gaps and jitter

After projection, a typical player might have pitch positions in frames 0, 1, 2, 5, 6, 10, 11... with gaps at 3, 4, 7, 8, 9. The detector missed them (blur, occlusion, being out of frame). Playback at 25 fps needs a position for every single frame, so we have to fill in the missing ones.

We also have a noise problem. Even on frames where the detector did fire, the bounding box wobbles slightly from frame to frame — the player is standing still but the box shifts 2–3 pixels because of detector variance. Plotted as a trajectory this looks like vibration. A smoothing filter removes it.

The pipeline applies three steps in order: linear interpolation, Savitzky-Golay smoothing, velocity clamping.

---

### Step 1 — Linear interpolation

Think of it like connecting dots on a graph with straight lines. If a player was at pitch position (400, 300) in frame 10 and (420, 340) in frame 15, we assume they moved in a straight line between those two frames and fill in:

```
frame 11 → (404, 308)
frame 12 → (408, 316)
frame 13 → (412, 324)
frame 14 → (416, 332)
```

Each step is (target - start) / number of steps. This is what `np.interp` does — it takes your known x-coordinates (frame numbers) and known y-values (positions) and fills in a value for every frame between the first and last detection.

Note: frames before the first detection and after the last detection are not filled in at all. If track 42 first appears in frame 10 and disappears in frame 80, we produce positions for frames 10–80 only.

---

### Step 2 — Savitzky-Golay smoothing

Linear interpolation fills gaps, but it does not fix the jitter on detected frames. Savitzky-Golay (SG) filtering is a sliding-window polynomial fit. The plain English version: for each frame, look at the N frames around it, fit a smooth curve through them, and replace the current value with the point on that curve.

The key parameter is the window size — how many surrounding frames to consider.

| Track length | Window |
|---|---|
| More than 20 frames | 15 frames |
| 10 to 20 frames | 11 frames |
| Fewer than 10 frames | no smoothing |

Why different windows? A large window smooths more aggressively. For short tracks (say, 8 frames) a window of 15 would be wider than the whole track — there are not enough data points to fit reliably, and SG filtering can introduce artefacts near the edges of short arrays. Very short tracks are left alone.

The window must always be an odd number (a quirk of how polynomial fits work with symmetric windows). If the computed value is even, it is decremented by 1.

Important: SG smoothing is applied to the full sequence — detected frames and interpolated frames together. An earlier version only smoothed the interpolated gaps and left detected frames as raw values. This caused a visible stutter: every time playback hit a detected frame, the position jumped because it had not been smoothed. Applying SG to everything makes the motion continuous.

---

### Step 3 — `_apply_max_velocity` — clamping runaway jumps

After smoothing, there can still be outlier jumps. For example, if the tracker assigned the same track ID to two different players (a known BotSort failure mode), the position can jump from one side of the pitch to the other in a single frame.

The maximum realistic speed for a GAA player is roughly 10 m/s. At 10 px/m and 25 fps that is:

```
10 m/s × 10 px/m ÷ 25 fps = 4.0 px per frame
```

So `_DEFAULT_MAX_VEL_PX = 4.0`.

The algorithm walks through the position sequence frame by frame. When a step exceeds 4 px, it does not discard the point — it moves it along the same direction but caps the distance:

```
For each frame i:
    dist = distance from position[i-1] to position[i]
    if dist > 4.0:
        scale = 4.0 / dist
        position[i] = position[i-1] + (position[i] - position[i-1]) * scale
```

The phrase "moves in the same direction but capped" means: if a player jumped 20 px to the right, we move them 4 px to the right instead. We do not snap them back.

Crucially, this correction is not carried forward. Position[i+1] is compared against the corrected position[i], so if the next frame is fine, only the one bad frame was adjusted. This prevents a chain reaction where fixing one frame pulls all subsequent positions toward it.

---

### `interpolate_trajectories` — putting it all together

The entry point for trajectory processing. It handles one track at a time and applies the three steps above.

**Full sequence of steps:**

1. Filter: keep only positions within the requested `[start_frame, end_frame]` window. Clips where the video was trimmed.

2. Skip single-point tracks: a track with only one detected frame has nothing to interpolate between. Return it as-is.

3. Linear interpolation: fill every frame between first and last detection.

4. Canvas clip: after interpolation, clamp all positions to the canvas boundary (0 to 850 in x, 0 to 1400 in y). Extrapolation artefacts or projection errors can occasionally push a point outside the canvas.

5. Savitzky-Golay smoothing (if the track is long enough): smooth the full sequence. Clip to canvas bounds again because SG can introduce slight overshoots at the edges.

6. Velocity clamping: apply `_apply_max_velocity`, then clip once more.

7. Label each output position: frames that were detected get their original source label (`"homography"` or `"homography_interp"`). Frames that were filled in get `source = "interpolated"`.

Output is a flat list of positions sorted by `(frame_idx, track_id)`.
