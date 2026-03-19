# App Endpoints (`app.py`)

FastAPI application entry point. Declares all HTTP endpoints, helper functions, middleware, and startup/shutdown logic.

---

## Application Setup

### `lifespan(app)`
`@asynccontextmanager` used by FastAPI for startup/shutdown logic.

- Creates `data/videos/`, `data/tracks/`, `data/annotations/` directories if they do not exist.
- Calls `_restore_videos_from_disk()` to repopulate the in-memory store on restart.
- Yields (application runs).

### CORS Middleware
Reads allowed origins from the `ALLOWED_ORIGINS` environment variable (comma-separated, defaults to `"*"`). Allows all methods and headers.

### Environment Variables

| Variable | Default | Meaning |
|----------|---------|---------|
| `MAX_VIDEO_SIZE_MB` | `500` | Maximum upload size |
| `ALLOWED_ORIGINS` | `"*"` | CORS allow-list |
| `DATA_DIR` | `"data"` | Root for all persisted files |

---

## Helper Functions

### `_restore_videos_from_disk() → None`
Scans `VIDEOS_DIR` for `*_meta.json` files and repopulates `store.videos`. Called once at startup. Skips entries whose video file no longer exists. Logs the number of restored videos.

### `_get_video_or_404(video_id) → dict`
Looks up `video_id` in `store.videos`. Raises `HTTPException(404)` if not found. Returns the metadata dict on success. Used by every endpoint that needs an existing video.

### `_save_json(path, data) → None` / `_load_json(path) → Optional[dict]`
Generic JSON persistence helpers. `_load_json` returns `None` if the file does not exist.

### `_serialize_H(h_dict) → dict` / `_deserialize_H(data) → dict`
Convert `Dict[int, np.ndarray]` ↔ `Dict[str, list]` for JSON serialisation. `numpy` arrays cannot be serialised directly; `tolist()` produces nested Python lists. On load, `np.array(v)` reconstructs the arrays.

### `validate_video_upload(file, content) → None`
Raises `HTTPException` if:
- File size exceeds `MAX_VIDEO_SIZE` (413 Too Large)
- Filename does not end in `.mp4` (400)
- Content-Type is not `video/mp4` or `application/octet-stream` (400)

### `save_video_meta / load_detections / save_detections`
Thin wrappers around `_save_json` / `_load_json` for specific file paths.

### `_save_homography_dict(video_id, key, h_dict) → None`
Saves `{video_id}_{key}.json` to `ANNOTATIONS_DIR`. The `key` is either `"v3_anchor_homographies"` or `"v3_homographies"`.

### `_load_homography_dict(video_id, key) → Optional[Dict[int, np.ndarray]]`
Loads the corresponding JSON file and deserialises to a numpy dict. Returns `None` if the file does not exist.

### `save_annotations / load_annotations`
Persists/loads the user's keypoint and line annotations per frame to `{video_id}_annotations.json`. `_serialise_ann_value` handles both Pydantic model instances and plain dicts.

### `_resolve_homography(video_id, frame_idx) → Optional[np.ndarray]`
Returns the per-frame v3 homography for `frame_idx`. Tries memory cache first, then disk. If the exact frame is not present, returns the H for the nearest frame by absolute index distance (nearest-anchor fallback). Returns `None` if no homographies exist at all.

### `_draw_reference_lines(warped) → None`
Draws semi-transparent pitch reference lines (opacity `_LINE_ALPHA = 0.45`) onto a warped canvas image **in place** using `cv2.addWeighted`.

Lines drawn:
- 9 dashed horizontal lines (13m, 20m, 45m, 65m, halfway, and their mirrors)
- 2 semicircles at the 20m lines (radius 13m = 130 px)
- 4 vertical lines for the 13m box (x=33m, x=52m from each endline to the 13m line)
- 6 lines for the small (goalie) box (x=35.5m–49.5m, depth=4.5m from each endline)

Each line's pixel position is computed by `y_px = int(y_m / GAA_PITCH_LENGTH * OUT_H)`.

---

## Endpoints

### `GET /health`
Returns `{"status": "ok"}`. Used for deployment health checks.

---

### `POST /videos` → `VideoCreateResponse`
Upload a video file.

1. Reads full file content into memory.
2. Calls `validate_video_upload` (size, extension, content-type checks).
3. Assigns a UUID, saves file to `VIDEOS_DIR/{id}.mp4`.
4. Calls `get_video_metadata` (OpenCV) to extract fps, num_frames, width, height.
5. Stores metadata in `store.videos` and persists to `{id}_meta.json`.
6. Returns `VideoCreateResponse`.

Error handling: if metadata extraction fails, the uploaded file is deleted before raising 400.

---

### `GET /videos/{video_id}/frame/{frame_idx}` → JPEG
Extract a single raw frame.

- Validates frame index is within bounds.
- Calls `extract_frame(video_path, frame_idx)` using OpenCV seek.
- Returns JPEG bytes with `Cache-Control: max-age=3600`.

---

### `GET /videos/{video_id}/frames/{frame_idx}/warped` → JPEG
Return a bird's-eye view of the pitch for a given frame.

Query params:
- `players` (bool, default `False`) — if true, overlay player positions as coloured dots with track ID labels.

1. Calls `_resolve_homography` to get the best available H (v3 per-frame, nearest-anchor fallback).
2. Reads the frame directly via `cv2.VideoCapture.set(CAP_PROP_POS_FRAMES, ...)`.
3. Calls `warp_frame(frame, H, OUT_W, OUT_H)` → `cv2.warpPerspective`.
4. Calls `_draw_reference_lines(warped)` for pitch overlay.
5. If `players=true`, draws a circle + track ID for every `PlayerPitchPosition` at `frame_idx`.
6. Encodes as JPEG at quality 85.
7. Cache-Control: `no-cache` if players, `max-age=300` otherwise.

Player dot colours alternate red/blue based on `track_id % 2`.

---

### `GET /videos/{video_id}/frames/{frame_idx}/detections_overlay` → JPEG
Raw video frame with BotSort bounding boxes overlaid.

Loads detections from cache or disk. For each detection at `frame_idx`, draws a coloured rectangle + label badge. Colour is determined by `hue = (track_id * 137.508) % 180` (golden-angle HSV, same scheme as the frontend pitch canvas). Returns JPEG at quality 85, cached 1 hour.

---

### `GET /videos/{video_id}/detections` → `List[Detection]`
Return all YOLO+BotSort detections. Tries memory cache first, then disk. Raises 404 if not found.

---

### `POST /videos/{video_id}/track` → `TrackResponse`
Run YOLO+BotSort tracking.

- If detections already exist on disk, loads them (idempotent — won't re-run tracking).
- Otherwise, calls `run_tracking(video_path)` (lazily imported to avoid loading heavy ML dependencies at startup).
- Caches in `store.detections_cache` and persists to disk.
- Returns `frames_processed` (last frame index + 1) and `tracks` (unique track count).

---

### `POST /videos/{video_id}/homographies/v3` → dict
Compute anchor homographies and propagate per-frame. The most computationally expensive endpoint.

Query params (all optional with defaults):

| Param | Default | Description |
|-------|---------|-------------|
| `num_samples_per_line` | `10` | Points sampled along each line annotation |
| `ransac_iterations` | `2000` | Max RANSAC trials for keypoint-only H₀ |
| `ransac_threshold` | `5.0` | RANSAC inlier threshold (canvas pixels) |
| `keypoint_weight` | `20.0` | Weight ratio keypoints vs line samples |

Steps:
1. Builds `annotations_dict` from the request body.
2. Runs `compute_homographies_with_lines_v3` in a thread (`asyncio.to_thread`) to avoid blocking the event loop.
3. If no valid homographies, returns 400 with per-frame error details.
4. Saves anchor Hs to cache + disk.
5. Runs `build_optical_flow_per_frame_H` in a thread.
6. Falls back to anchor Hs if optical flow propagation fails.
7. Saves per-frame Hs to cache + disk.
8. Returns `{frames, per_frame_count, info}`.

---

### `GET /line-constraints/available-lines` → dict
Returns the `GAA_PITCH_LINES` dict (line IDs → Y meters) plus human-readable descriptions for each.

---

### `POST /videos/{video_id}/map_players` → `List[PlayerPitchPosition]`
Map player detections to pitch canvas coordinates.

1. Loads detections from disk.
2. Loads per-frame Hs from cache or disk (v3 only).
3. Calls `filter_detections_for_mapping` (drops ball and referee detections).
4. Calls `map_players_to_pitch` — applies each detection's frame H to the bottom-centre of the bounding box.
5. Stores result in `store.player_positions_cache`.

---

### `GET /videos/{video_id}/homographies/anchor-quality` → dict
Compute per-keypoint reprojection quality report for all anchor frames.

For each anchor frame:
1. Loads saved annotations and anchor Hs.
2. For each annotated keypoint, projects it through H and computes Euclidean distance to the expected canvas coordinate.
3. Labels each point: `error < 15px` → "good", `15–30px` → "high", `>30px` → "outlier".
4. Labels impact: "helpful", "marginal", or "harmful".
5. Computes per-frame summary: mean error, max error, outlier count, overall quality ("good"/"warning"/"bad"), and recommendation text.

The response `{"anchors": [...]}` is consumed by `PipelineSteps.tsx` to show a per-anchor quality table with colour-coded reprojection errors.

---

### `POST /videos/{video_id}/interpolate` → `InterpolationResponse`
Interpolate and smooth player trajectories.

Query params:
- `start_frame`, `end_frame` — frame range (inclusive)
- `sg_long_window` (default 15) — SG window for tracks > 20 frames
- `sg_mid_window` (default 11) — SG window for tracks 10–20 frames
- `max_vel_px` (default 4.0) — max displacement per frame (0 = disabled)

1. Filters `player_positions_cache` to `source in ("homography", "homography_interp")`.
2. Calls `interpolate_trajectories`.
3. Merges the new dense positions with positions outside the requested range (preserving them).
4. Stores merged result in `store.player_positions_cache`.

---

### `GET /videos/{video_id}/players` → `List[PlayerPitchPosition]`
Return all player positions (sparse + interpolated), sorted by `(frame_idx, track_id)`. Raises 404 if no positions exist.

---

### `POST /videos/{video_id}/classify-teams` → dict
Classify player tracks as `'ellistown'` or `'opposition'` using jersey colour analysis.

1. Loads detections from cache or disk. Raises 400 if no detections exist (run tracking first).
2. Filters detections to player tracks only via `filter_detections_for_mapping`.
3. Calls `classify_tracks(video_path, player_detections)` in a thread (`asyncio.to_thread`). This samples up to 30 frames per track, extracts the jersey HSV colour from the top 50% of each bounding box, and classifies based on Ellistown yellow fraction (see `TEAM_CLASSIFIER.md`).
4. Stores result in `store.team_classifications_cache` and persists to `{video_id}_team_classifications.json`.
5. Computes a summary:
   - `num_ellistown`, `num_opposition` — track counts per team.
   - `num_referee` — always 0 (referees are filtered before classification).
   - `mean_confidence` — average classification confidence across all tracks.
   - `low_confidence_tracks` — list of track IDs with `confidence < 0.6`.
   - `hsv_cluster_separation` — Euclidean norm of the difference between the mean HSV of Ellistown tracks and opposition tracks (`None` if either group is empty).
6. Returns `{"classifications": {str(track_id): {team, confidence, mean_hsv}}, "summary": {...}}`.

---

### `GET /videos/{video_id}/classify-teams` → dict
Return stored team classifications for a video. Tries memory cache first, then disk. Raises 404 if no classifications exist. Returns `{"classifications": {str(track_id): {team, confidence, mean_hsv}}}`.

---

### `POST /videos/{video_id}/classify-teams` body / `PATCH /videos/{video_id}/classify-teams` → dict
Override the team assignment for a single track.

Request body (`TeamOverrideRequest`):
| Field | Type | Description |
|-------|------|-------------|
| `track_id` | `int` | The track to reassign |
| `team` | `str` | One of `"ellistown"`, `"opposition"`, `"referee"`, `"ignore"` |

Raises 400 if `team` is not in `VALID_TEAMS`. Merges the override into the existing classification dict, persists to disk, and returns the full updated `{"classifications": {...}}` dict.
