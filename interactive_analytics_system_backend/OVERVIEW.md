# Backend Overview

The backend is a FastAPI application that accepts an uploaded GAA football video, runs YOLO+BotSort tracking to detect and identify players across frames, computes perspective-correcting homographies from user-supplied pitch annotations, maps every player detection to a fixed 2D pitch canvas, then interpolates and smoothes the resulting trajectories for playback in the frontend.

---

## Full Data-Flow

```
1. Upload MP4
   POST /videos
   → save to disk, extract metadata (fps, num_frames, width, height)
   → assign UUID, store metadata in memory + JSON file

2. Track
   POST /videos/{id}/track
   → run YOLO+BotSort (remote GPU via Modal or local CPU fallback)
   → produce List[Detection]  (frame_idx, track_id, bbox, confidence, class)
   → persist to TRACKS_DIR/{id}.json

3. Annotate (in browser, no backend endpoint)
   User clicks pitch keypoints and line segments on each anchor frame.

4. Compute Homographies  (v3 endpoint)
   POST /videos/{id}/homographies/v3
   → for each annotated anchor frame:
       a. RANSAC H₀ from keypoints only
       b. Weighted DLT (Hartley-normalised) adds line constraints
       c. SVD → denormalise → sanity-check fallback to H₀ if degenerate
   → persist anchor_Hs
   → run build_optical_flow_per_frame_H:
       Phase 1: LK forward-backward optical flow for every consecutive pair
       Phase 2: chain H[t] = H[t-1] @ inv(OF_H), linear drift correction per segment
       Phase 3: Savitzky-Golay smoothing per H element, re-pin anchors
   → persist per_frame_Hs

5. Map Players
   POST /videos/{id}/map_players
   → filter out ball detections + referee tracks
   → for each detection, apply H[frame_idx] to bottom-centre of bbox
   → produce List[PlayerPitchPosition] in pitch-canvas pixel coords
   → store in memory

6. Interpolate
   POST /videos/{id}/interpolate
   → per track: linear interp → SG smooth → max-vel clamp → canvas clip
   → produce dense List[PlayerPitchPosition] for the requested frame range
   → merge with existing positions, store in memory

7. Playback
   GET /videos/{id}/players          → all positions (sparse + interpolated)
   GET /videos/{id}/frames/{f}/warped → warped JPEG + pitch reference lines
```

---

## Coordinate Systems

| Space | Description | Range |
|-------|-------------|-------|
| **Image pixels** | Camera frame (x right, y down) | 0..width × 0..height (e.g. 1920×1080) |
| **Pitch-canvas pixels** | Fixed output canvas (x right, y down) | 0..850 × 0..1400 |
| **Pitch meters** | Real-world GAA pitch (x right, y "away") | 0..85 m × 0..140 m |

Conversion from meters to canvas pixels is trivial: scale by `OUT_W / GAA_PITCH_WIDTH` and `OUT_H / GAA_PITCH_LENGTH` (both equal exactly **10 px/m**).

Homographies map directly from **image pixels → pitch-canvas pixels**. Meter values are only used to look up destination points when building the DLT system; they never appear at runtime after setup.

---

## Endpoint Table

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Liveness check |
| POST | `/videos` | Upload MP4, extract metadata |
| GET | `/videos/{id}/frame/{idx}` | Raw frame as JPEG |
| GET | `/videos/{id}/frames/{idx}/warped` | Warped + pitch reference lines; `?players=true` adds dots |
| GET | `/videos/{id}/frames/{idx}/detections_overlay` | Raw frame + BotSort bounding boxes |
| GET | `/videos/{id}/detections` | All raw detections |
| POST | `/videos/{id}/track` | Run YOLO+BotSort |
| POST | `/videos/{id}/homographies/v3` | Compute anchor Hs + propagate per-frame |
| GET | `/line-constraints/available-lines` | Line IDs usable for annotations |
| POST | `/videos/{id}/map_players` | Map detections → pitch coords |
| GET | `/videos/{id}/homographies/anchor-quality` | Per-keypoint reprojection quality report |
| POST | `/videos/{id}/interpolate` | Interpolate + smooth trajectories |
| GET | `/videos/{id}/players` | All player positions (sparse + interpolated) |
| POST | `/videos/{id}/classify-teams` | Classify tracks as Ellistown/opposition by jersey colour |
| GET | `/videos/{id}/classify-teams` | Return stored team classifications |
| PATCH | `/videos/{id}/classify-teams` | Override a single track's team assignment |

---

## In-Memory Store Layout (`store.py`)

```python
class VideoStore:
    videos:                      Dict[str, dict]                       # video metadata keyed by UUID
    detections_cache:            Dict[str, List[Detection]]            # raw YOLO detections
    v3_anchor_H_cache:           Dict[str, Dict[int, np.ndarray]]      # anchor frame Hs (keyed by frame_idx)
    v3_per_frame_H_cache:        Dict[str, Dict[int, np.ndarray]]      # propagated per-frame Hs
    player_positions_cache:      Dict[str, List[PlayerPitchPosition]]  # mapped + interpolated positions
    team_classifications_cache:  Dict[str, Dict[int, dict]]            # jersey-colour team classifications
```

All six dicts are keyed by `video_id` (UUID string). On restart the `videos` dict is repopulated from disk (`_restore_videos_from_disk`); the other caches start empty and are lazily reloaded from disk when needed.

---

## Disk Layout

```
data/
  videos/
    {id}.mp4            ← uploaded video file
    {id}_meta.json      ← fps, num_frames, width, height, duration_seconds
  tracks/
    {id}.json           ← list of Detection dicts
  annotations/
    {id}_annotations.json              ← user keypoints + line annotations per frame
    {id}_v3_anchor_homographies.json   ← anchor Hs (str(frame_idx) → 3×3 list)
    {id}_v3_homographies.json          ← per-frame Hs (str(frame_idx) → 3×3 list)
    {id}_team_classifications.json     ← jersey-colour classifications (str(track_id) → {team, confidence, mean_hsv})
```
