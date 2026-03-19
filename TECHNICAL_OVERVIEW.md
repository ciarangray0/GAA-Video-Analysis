# GAA Video Analysis — Technical Overview

A full-stack video analysis system for GAA (Gaelic football) footage. Given an uploaded MP4, the system detects and tracks players frame-by-frame using a custom YOLO model, computes a perspective-correcting homography for every frame of the video, and produces a 2D bird's-eye trajectory map of every player on a fixed pitch canvas. A Next.js frontend guides the user through the pipeline interactively.

---

## Table of Contents

1. [Repository Layout](#1-repository-layout)
2. [Architecture Overview](#2-architecture-overview)
3. [Coordinate Systems](#3-coordinate-systems)
4. [Pipeline Components](#4-pipeline-components)
   - 4.1 [Video Ingestion](#41-video-ingestion)
   - 4.2 [Player Detection & Tracking (YOLO + BotSort)](#42-player-detection--tracking-yolo--botsort)
   - 4.3 [GPU Inference (Modal)](#43-gpu-inference-modal)
   - 4.4 [Anchor Frame Annotation (Frontend)](#44-anchor-frame-annotation-frontend)
   - 4.5 [Homography Computation (v3)](#45-homography-computation-v3)
   - 4.6 [Per-Frame Propagation (Optical Flow)](#46-per-frame-propagation-optical-flow)
   - 4.7 [Player Mapping](#47-player-mapping)
   - 4.8 [Trajectory Interpolation & Smoothing](#48-trajectory-interpolation--smoothing)
   - 4.9 [Results Visualisation (Frontend)](#49-results-visualisation-frontend)
5. [Data Flow](#5-data-flow)
6. [In-Memory Store & Persistence](#6-in-memory-store--persistence)
7. [API Endpoint Reference](#7-api-endpoint-reference)
8. [Key Libraries & Dependencies](#8-key-libraries--dependencies)
9. [Configuration & Environment Variables](#9-configuration--environment-variables)
10. [Frontend Architecture](#10-frontend-architecture)
11. [Pitch Geometry Reference](#11-pitch-geometry-reference)
12. [Known Limitations & TODOs](#12-known-limitations--todos)
13. [Development History — What Was Tried, What Worked, What Didn't](#13-development-history--what-was-tried-what-worked-what-didnt)
14. [Getting Started for a New Engineer](#14-getting-started-for-a-new-engineer)

---

## 1. Repository Layout

```
GAA-Video-Analysis/
├── interactive_analytics_system_backend/
│   ├── app.py                         ← FastAPI application, all HTTP endpoints
│   ├── store.py                       ← In-memory state container
│   ├── pipeline/
│   │   ├── config.py                  ← Canvas size constants, model path
│   │   ├── gaa_pitch_config.py        ← Pitch geometry: vertices, lines, sidelines
│   │   ├── schemas.py                 ← Pydantic models for all data types
│   │   ├── video.py                   ← OpenCV wrappers: metadata + frame extraction
│   │   ├── rendering.py               ← warp_frame (cv2.warpPerspective wrapper)
│   │   ├── detect.py                  ← run_tracking: dispatches local or remote GPU
│   │   ├── homography.py              ← Anchor H computation (v3 DLT algorithm)
│   │   ├── line_constraints.py        ← sample_points_on_line, re-exports line dicts
│   │   ├── constrained_homography.py  ← Per-frame H propagation via LK optical flow
│   │   ├── map_players.py             ← Filter detections, map bbox → pitch coords
│   │   └── trajectories.py            ← Linear interp → SG smooth → velocity clamp
│   └── gpu_inference/
│       ├── __init__.py                ← GPUInferenceClient (HTTP client for Modal)
│       └── modal_yolo.py              ← Modal serverless GPU service definition
│
├── interactive_analytics_system_frontend/
│   ├── pages/
│   │   └── index.tsx                  ← Root page, all cross-step state
│   ├── components/
│   │   ├── VideoUploader.tsx
│   │   ├── AnchorFrameAnnotator.tsx   ← Frame + pitch canvas annotation UI
│   │   ├── PipelineSteps.tsx          ← Steps A–D runner + results display
│   │   ├── ResultsViewer.tsx          ← Side-by-side video + 2D pitch playback
│   │   └── DebugLog.tsx               ← API activity log sidebar
│   ├── lib/
│   │   ├── api.ts                     ← All backend API call functions
│   │   ├── pitch.ts                   ← Canvas drawing: pitch diagram + results view
│   │   └── constants.ts               ← Pitch vertices, line segments, display scale
│   └── types/
│       └── index.ts                   ← TypeScript interfaces
│
├── pipeline_testing_and_research/     ← DO NOT MODIFY (diagnostic scripts + test output)
├── CLAUDE.md                          ← AI assistant context file
└── TECHNICAL_OVERVIEW.md              ← This file
```

---

## 2. Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     Browser (Next.js)                        │
│                                                             │
│  [1] Upload MP4 ──► [2] Configure anchors                   │
│       ▼                     ▼                               │
│  [3] Annotate frames    [4] Run Pipeline (A→B→C→D)          │
│       │                     │                               │
│       │  POST /videos/{id}/homographies/v3                  │
│       │  POST /videos/{id}/track                            │
│       │  POST /videos/{id}/map_players                      │
│       │  POST /videos/{id}/interpolate                      │
│       │                     │                               │
│  [5] Results Viewer ◄────────┘                              │
└─────────────────────────────────────────────────────────────┘
              │ HTTP (REST + JSON)
              ▼
┌─────────────────────────────────────────────────────────────┐
│               FastAPI Backend (Render / local)               │
│                                                             │
│  VideoStore (in-memory)                                     │
│    └── videos, detections, anchor_Hs, per_frame_Hs,         │
│        player_positions                                     │
│                                                             │
│  Pipeline modules (pure Python/NumPy/OpenCV/SciPy)          │
│    detect → homography → constrained_homography →           │
│    map_players → trajectories                               │
└─────────────────────────────────────────────────────────────┘
              │ HTTP (base64 video → JSON detections)
              ▼
┌─────────────────────────────────────────────────────────────┐
│            Modal Serverless GPU (T4)                         │
│   YOLOv8-small + BotSort, imgsz=960, conf=0.35              │
└─────────────────────────────────────────────────────────────┘
```

The backend is stateless between restarts (except for files on disk). The in-memory store is repopulated from disk at startup via `_restore_videos_from_disk`. Heavy ML inference is offloaded to Modal to avoid needing a GPU on the backend server.

---

## 3. Coordinate Systems

Understanding the three coordinate spaces is critical for working on this codebase. Confusion between them has caused the most significant bugs to date.

### Image Pixels (Camera Space)
- Origin: top-left of the video frame.
- x right, y down.
- Range: `0..video_width` × `0..video_height` (e.g. 1920×1080).
- Used in: `Detection` bboxes, annotation `PitchPoint.x_img/y_img`, LK optical flow.

### Pitch-Canvas Pixels
- Origin: top-left corner of the top goal endline.
- x right (east), y down (toward the bottom goal).
- Fixed size: **850 × 1400 px** (`OUT_W × OUT_H`).
- Used in: `PlayerPitchPosition.x_pitch/y_pitch`, all H matrix outputs, `warp_frame` output.

### Pitch Meters
- Same orientation as pitch-canvas pixels.
- GAA pitch: **85 m wide × 140 m long**.
- Range: x ∈ [0, 85], y ∈ [0, 140].
- Used only when looking up named vertex positions or setting up DLT destination points.
- Conversion: exactly **10 px/m** in both dimensions (`OUT_W/85 = OUT_H/140 = 10`).

### Conversion Formula
```python
x_canvas_px = x_meters * 10    # or: x_m / GAA_PITCH_WIDTH * OUT_W
y_canvas_px = y_meters * 10    # or: y_m / GAA_PITCH_LENGTH * OUT_H
```

### Homography Direction
All H matrices map: `image_pixels → pitch_canvas_pixels`. Applied as:
```python
p_canvas = H @ [x_img, y_img, 1.0]
p_canvas /= p_canvas[2]          # perspective division
x_canvas, y_canvas = p_canvas[0], p_canvas[1]
```

### Frontend Display Scale
The frontend pitch canvas is displayed at 40% scale:
- `PITCH_DISPLAY_WIDTH  = 850  × 0.4 = 340 px`
- `PITCH_DISPLAY_HEIGHT = 1400 × 0.4 = 560 px`

Player positions from the backend are in 850×1400 space. The frontend scales them:
```typescript
x_display = (x_pitch / PITCH_CANVAS_W) * PITCH_DISPLAY_WIDTH
y_display = (y_pitch / PITCH_CANVAS_H) * PITCH_DISPLAY_HEIGHT
```

---

## 4. Pipeline Components

### 4.1 Video Ingestion

**File:** `pipeline/video.py`, `app.py`

On `POST /videos`, the backend:
1. Validates the file (size ≤ 500MB, `.mp4` extension, correct MIME type).
2. Assigns a UUID and saves to `data/videos/{uuid}.mp4`.
3. Extracts metadata using OpenCV: fps, num_frames, width, height.
4. Persists metadata to `{uuid}_meta.json` and stores in `store.videos`.

`extract_frame(video_path, frame_idx)` uses `cv2.VideoCapture.set(CAP_PROP_POS_FRAMES, idx)` to seek and returns JPEG bytes at quality 85.

---

### 4.2 Player Detection & Tracking (YOLO + BotSort)

**File:** `pipeline/detect.py`

On `POST /videos/{id}/track`, the backend runs `run_tracking(video_path)`. This is **idempotent** — if detections already exist on disk, they are loaded rather than recomputed.

**Model:** Custom YOLOv8-small (`v8s_960_v9.pt`) trained at 960px input on GAA footage. Three detection classes:
- `"GAA-player-lablers"` — players
- `"Ball-labelers"` — the ball
- `"Refree-lablers"` — referees (note: typos are intentional — they match the model's training labels)

**Tracker:** BotSort (Ultralytics implementation), configured via `botsort.yaml`. Provides persistent `track_id` values across frames, even when a player is briefly occluded.

**Dispatch:**
- `GPU_PROVIDER=modal` → sends base64-encoded video to the Modal HTTP endpoint.
- `GPU_PROVIDER=local` (default) → runs YOLO locally on CPU (very slow, development only).

**Output:** `List[Detection]` with fields: `frame_idx`, `track_id`, `x1, y1, x2, y2`, `confidence`, `class_name`. Saved to `data/tracks/{uuid}.json`.

---

### 4.3 GPU Inference (Modal)

**Files:** `gpu_inference/modal_yolo.py`, `gpu_inference/__init__.py`

**Modal service (`modal_yolo.py`):**
- Defines a `YOLOTracker` class decorated with `@app.cls(gpu="T4", timeout=600)`.
- Container image: Debian slim + Python 3.11 + system OpenCV deps + ultralytics + torch.
- Model weights stored in a Modal Volume (`yolo-model-cache`) mounted at `/model_cache`.
- `load_model()` (`@modal.enter()`) — runs once per container cold-start.
- `_run_tracking(video_bytes)` — writes bytes to a temp file, runs `model.track(stream=True)`, returns detection dicts.
- `track_video_endpoint()` (`@modal.fastapi_endpoint(POST)`) — decodes base64, calls `_run_tracking`, returns JSON.

**Client (`__init__.py`):**
- `GPUInferenceClient` wraps `httpx.Client` with a 600-second timeout.
- `_track_modal(video_path)` reads the video file, base64-encodes it, POSTs to the Modal endpoint URL, converts response dicts back to `Detection` objects.
- `get_gpu_client()` returns a module-level singleton; created on first call from `GPU_PROVIDER` and `GPU_ENDPOINT_URL` env vars.

**Setup:**
```bash
pip install modal
modal token new
modal volume put yolo-model-cache v8s_960_v9.pt /v8s_960_v9.pt
modal deploy gpu_inference/modal_yolo.py
# Copy the printed endpoint URL to GPU_ENDPOINT_URL env var
```

---

### 4.4 Anchor Frame Annotation (Frontend)

**File:** `components/AnchorFrameAnnotator.tsx`

The user selects anchor frames (e.g. one per second) and annotates each with correspondences between the video frame and a GAA pitch diagram. Two annotation modes:

#### Point Mode
1. User clicks a feature in the video frame (e.g. a corner flag).
2. A crosshair appears at the click location.
3. User clicks the corresponding location in the pitch diagram.
4. System snaps to the nearest named vertex (within 20px) or nearest pitch line segment (within 15px), encoding the location as a `pitch_id`.
5. A `PitchPoint` is created: `{pitch_id, x_img, y_img}` where `x_img/y_img` are **original image pixels**.

#### Line Mode
1. User selects a pitch line (e.g. "45m Line (Top)") from a dropdown.
2. User clicks two points that both lie on that line in the video frame.
3. A `LineAnnotation` is created: `{line_id, u1, v1, u2, v2}`.
4. The backend samples `N` points along this segment and adds them as 1D constraints in the DLT system.

**Key implementation detail — the outline fix:**
The canvas element uses CSS `outline: 2px` (not `border: 2px`). `getBoundingClientRect()` returns the border-box, so a 2px border would shift `rect.left/top` by 2px, causing a systematic ~4–8px offset in image-space click coordinates. `outline` is drawn outside the layout box and does not affect `getBoundingClientRect`.

**Coordinate conversion:**
```typescript
const x = (e.clientX - rect.left) * img.naturalWidth  / rect.width
const y = (e.clientY - rect.top)  * img.naturalHeight / rect.height
```
This works at any zoom level because `rect.width = canvas.width * zoom`.

**Persistence:** Annotations are auto-saved to `localStorage` on every change under `"gaa_annotations_{videoFilename}"`. Restored when the user regenerates anchor frames for the same video.

---

### 4.5 Homography Computation (v3)

**File:** `pipeline/homography.py`, endpoint `POST /videos/{id}/homographies/v3`

Computes a 3×3 perspective homography matrix for each annotated anchor frame.

#### Algorithm: Weighted DLT with Hartley Normalisation

For each anchor frame with ≥ 4 keypoints:

**Step 1 — RANSAC keypoint-only H (H₀)**
```python
H0, _ = cv2.findHomography(pts_image, pts_canvas, cv2.RANSAC, threshold=5.0, maxIters=2000)
```
This is the primary robust estimate. RANSAC handles misannotated keypoints. If no line annotations are present, H₀ is used directly.

**Step 2 — Hartley Normalisation**

*Why this is mandatory:* Without normalisation, the DLT matrix contains products of image coordinates (~0–1920) and canvas coordinates (~0–1400). These reach ~10⁶, making the SVD numerically unstable. The result is a catastrophically wrong H that maps the entire pitch to a small cluster.

Hartley normalisation transforms both point sets so their centroid is the origin and mean distance from origin is √2:
```python
pts_image_n,  T_img    = _hartley_normalize(pts_image)
pts_canvas_n, T_canvas = _hartley_normalize(pts_canvas)
```

**Step 3 — Build Weighted DLT System**

For each keypoint correspondence `(u,v) → (x,y)` (in normalised coords), two rows:
```
[u, v, 1, 0, 0, 0, -x·u, -x·v, -x]  weight = keypoint_weight (default 20)
[0, 0, 0, u, v, 1, -y·u, -y·v, -y]  weight = keypoint_weight
```

For each horizontal line sample `(u,v) → known y_c`:
```
[0, 0, 0, u, v, 1, -y_c·u, -y_c·v, -y_c]  weight = 1.0
```

For each vertical sideline sample `(u,v) → known x_c`:
```
[u, v, 1, 0, 0, 0, -x_c·u, -x_c·v, -x_c]  weight = 1.0
```

**Why keypoint_weight=20?** With ~4 keypoints (8 rows × weight 20 = effective 160) vs ~30 line samples (30 rows × weight 1 = effective 30), the ratio is ~5:1. Keypoints dominate and lines only correct unconstrained directions (e.g. X-skew in midfield where no keypoints exist).

**Step 4 — Weighted SVD**
```python
A = np.array(rows)
_, _, Vt = np.linalg.svd(A * w_vec[:, np.newaxis], full_matrices=False)
H_norm = Vt[-1].reshape(3, 3)   # null vector = smallest singular value
```

**Step 5 — Denormalise**
```python
H = np.linalg.inv(T_canvas) @ H_norm @ T_img
H /= H[2, 2]
```

**Step 6 — Sanity Check**

Fall back to H₀ if any of:
- H contains NaN values
- `np.linalg.cond(H) > 1e8` (near-singular)
- Mean reprojection error of H is more than 2× that of H₀ (line constraints actively hurt)

**Quality reporting:** `_fill_info` computes per-keypoint errors, mean/max, coverage score (fraction of a 3×2 image grid covered by keypoints), and an overall "good"/"warning"/"bad" rating.

---

### 4.6 Per-Frame Propagation (Optical Flow)

**File:** `pipeline/constrained_homography.py`

After computing anchor Hs, `build_optical_flow_per_frame_H` propagates them to every frame.

#### Phase 1 — Inter-Frame Optical Flow

For every consecutive frame pair `(t, t+1)`, computes `H_{t→t+1}` via Lucas-Kanade optical flow (`_lk_inter_frame_H`):

1. **Feature detection:** `cv2.goodFeaturesToTrack` on the grayscale frame. A binary mask zeros the top 35% of the frame (sky, stands, advertising boards) — features there move independently of the pitch.
2. **Forward flow:** `cv2.calcOpticalFlowPyrLK(g1, g2, pts1)` — tracks features from frame `t` to `t+1`.
3. **Backward flow:** `cv2.calcOpticalFlowPyrLK(g2, g1, pts2)` — tracks found points back.
4. **Forward-backward filter:** keep only points where `|pts1 - pts1_back| < 1.0px`. This removes moving players — their flow is inconsistent (they appear in `g1` but their tracked position in `g2` does not track back to the same location in `g1`).
5. **Robust H:** `cv2.findHomography(RANSAC, threshold=3.0)` on surviving points. Needs ≥8 inliers.

#### Phase 2 — Chaining and Drift Correction

For each segment between anchors `(A, B)`:

**Forward chain:**
```python
H[t] = H[t-1] @ inv(of_Hs[t-1])   # t = A+1 .. B
```
`of_Hs[t-1]` maps frame `t-1` → frame `t`. The inverse maps `t` → `t-1`. Composing with `H[t-1]` gives `H[t]`: a mapping from frame `t` image coords to the pitch canvas.

**Drift correction:**
Chaining accumulates small errors. At anchor `B`, the chained estimate `H_chain[B]` may differ from the trusted `anchor_homographies[B]`. The drift matrix:
```python
H_drift = anchor_homographies[B] @ inv(H_chain[B])
```
...is blended linearly over the segment:
```python
alpha = (t - A) / (B - A)
H[t] = ((1-alpha)*I + alpha*H_drift) @ H_chain[t]
```
At `t=A`, no correction (alpha=0). At `t=B`, full correction (alpha=1, matches trusted anchor). Both anchors are then re-pinned exactly.

**Before first anchor / after last anchor:** the nearest anchor's H is used directly.

#### Phase 3 — Savitzky-Golay Smoothing

For each inter-anchor segment, each of the 9 H matrix elements is smoothed independently with `scipy.signal.savgol_filter(window=min(21, n_seg), polyorder=2)`. Anchor frames are re-pinned after smoothing. Segments shorter than 5 frames are not smoothed.

---

### 4.7 Player Mapping

**File:** `pipeline/map_players.py`

**Filtering (`filter_detections_for_mapping`):**
- Drop all detections with `class_name == CLASS_BALL`.
- Find all `track_id` values that have **any** detection classified as `CLASS_REFEREE`.
- Drop **all** detections for those track IDs (whole-track removal handles occasional misclassifications).

**Mapping (`map_players_to_pitch`):**

For each surviving detection, apply the per-frame H to the **bottom-centre** of the bounding box:
```python
x_foot = (det.x1 + det.x2) / 2   # horizontal centre
y_foot = det.y2                    # bottom edge of bbox
x_pitch, y_pitch = map_pixel_to_pitch(x_foot, y_foot, H[frame_idx])
```

Bottom-centre approximates where the player's feet contact the ground — the correct contact point for projecting a standing player through a ground-plane homography. Using the bbox centre or top would introduce systematic upward offset.

**Source labels:**
- `"homography"` — anchor frame (highest quality H)
- `"homography_interp"` — propagated frame (optical flow H)

---

### 4.8 Trajectory Interpolation & Smoothing

**File:** `pipeline/trajectories.py`

Per track (group of positions with the same `track_id`):

**Step 1 — Linear Interpolation**
```python
frames_track = np.arange(track_start, track_end + 1)
xs = np.interp(frames_track, known_frames, known_xs)
ys = np.interp(frames_track, known_frames, known_ys)
```
Fills every frame between first and last detection. No extrapolation beyond the track's range.

**Step 2 — Canvas Clip**
```python
xs = np.clip(xs, 0, OUT_W)
ys = np.clip(ys, 0, OUT_H)
```

**Step 3 — Savitzky-Golay Smoothing**

Window selection by track length:
| Track length | Window |
|-------------|--------|
| > 20 frames | `min(sg_long_window, n)`, default 15 |
| 10–20 frames | `min(sg_mid_window, n)`, default 11 |
| < 10 frames | None (no smoothing) |

Window must be odd. Applied to both x and y sequences independently with polynomial order 2.

**Critical:** Smoothing is applied to the full interpolated sequence including originally-detected frames. Earlier versions only smoothed filled gaps, leaving detected frames with raw (unsmoothed) values — this caused visible jitter at every anchor frame as the trajectory jumped between raw detections.

**Step 4 — Max-Velocity Clamping**
```python
for i in 1..n:
    dist = hypot(dx, dy)
    if dist > max_vel:                    # default 4.0 px/frame = 10 m/s at 25fps
        xs[i] = xs[i-1] + dx * (max_vel / dist)
        ys[i] = ys[i-1] + dy * (max_vel / dist)
```
Each violating step is scaled back; subsequent frames are relative to the corrected position (the correction is not propagated forward).

Default `max_vel_px = 4.0` corresponds to 10 m/s at 10 px/m and 25 fps — approximately the sprint speed limit in Gaelic football.

---

### 4.9 Results Visualisation (Frontend)

**File:** `components/ResultsViewer.tsx`, `lib/pitch.ts`

**Side-by-side layout:**
- Left: HTML `<video>` element using a blob URL created from the uploaded `File` object.
- Right: `<canvas>` redrawn on every frame change via `drawPitch`.

**Playback loop:**
```typescript
video.play().then(() => {
  animFrameRef.current = requestAnimationFrame(onPlaybackFrame)
})
```
`onPlaybackFrame` converts `video.currentTime` to a frame index and calls `onFrameChange`. The `.then()` chain ensures RAF only starts once the video is confirmed playing (avoids the canvas updating while the video is still loading).

**`drawPitch` (lib/pitch.ts):**
1. Draws the green pitch background + white markings (lines, semicircles, boxes).
2. Filters out tracks with fewer than 30 position entries (suppresses spurious short tracks).
3. Draws **ghost dots** (grey, semi-transparent) for tracks seen in past frames but absent now.
4. Draws **active player dots** coloured by `hsl((track_id × 137.508) % 360, 70%, 50%)` — the golden angle ensures maximally different hues for adjacent track IDs.

---

## 5. Data Flow

```
[Upload]  File → POST /videos
               → {video_id, fps, num_frames, ...}

[Track]   POST /videos/{id}/track
               → List[Detection] (frame_idx, track_id, bbox, class)
               → persisted to data/tracks/{id}.json

[Annotate] Browser → List[AnchorFrameAnnotation]
               → {frame_idx, points:[{pitch_id,x_img,y_img}], lines:[{line_id,u1,v1,u2,v2}]}

[Homography] POST /videos/{id}/homographies/v3
               → anchor_Hs: Dict[frame_idx → 3×3 H]  (saved as v3_anchor_homographies.json)
               → per_frame_Hs: Dict[frame_idx → 3×3 H] (saved as v3_homographies.json)

[Map]     POST /videos/{id}/map_players
               → List[PlayerPitchPosition] (frame_idx, track_id, x_pitch, y_pitch, source)
               → stored in store.player_positions_cache

[Interpolate] POST /videos/{id}/interpolate
               → additional PlayerPitchPosition objects with source="interpolated"
               → merged into store.player_positions_cache

[Fetch]   GET /videos/{id}/players
               → full List[PlayerPitchPosition] sorted by (frame_idx, track_id)

[Display] Frontend: drawPitch(canvas, positions, currentFrame)
```

---

## 6. In-Memory Store & Persistence

**File:** `store.py`

```python
class VideoStore:
    videos:                 Dict[str, dict]                     # video metadata
    detections_cache:       Dict[str, List[Detection]]          # YOLO detections
    v3_anchor_H_cache:      Dict[str, Dict[int, np.ndarray]]    # anchor frame Hs
    v3_per_frame_H_cache:   Dict[str, Dict[int, np.ndarray]]    # propagated per-frame Hs
    player_positions_cache: Dict[str, List[PlayerPitchPosition]] # mapped + interpolated
```

All dicts keyed by `video_id` (UUID string). The store is a module-level singleton (`store = VideoStore()`).

**On restart:** `_restore_videos_from_disk()` (called at lifespan startup) repopulates `store.videos` from `data/videos/*_meta.json`. Other caches start empty and are lazily loaded from disk in each endpoint when needed (e.g. `_load_homography_dict`).

**Disk files:**
```
data/videos/{id}.mp4
data/videos/{id}_meta.json
data/tracks/{id}.json
data/annotations/{id}_annotations.json
data/annotations/{id}_v3_anchor_homographies.json
data/annotations/{id}_v3_homographies.json
```

Homography matrices are serialised as `Dict[str(frame_idx), List[List[float]]]` (JSON requires string keys; numpy arrays serialise via `.tolist()`).

---

## 7. API Endpoint Reference

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/health` | Liveness check |
| POST | `/videos` | Upload MP4 |
| GET | `/videos/{id}/frame/{idx}` | Raw JPEG frame |
| GET | `/videos/{id}/frames/{idx}/warped` | Bird's-eye JPEG + pitch lines (`?players=true` adds dots) |
| GET | `/videos/{id}/frames/{idx}/detections_overlay` | Raw frame + BotSort bboxes |
| GET | `/videos/{id}/detections` | All raw detections |
| POST | `/videos/{id}/track` | Run YOLO+BotSort |
| POST | `/videos/{id}/homographies/v3` | Compute anchors + propagate per-frame |
| GET | `/line-constraints/available-lines` | Available line IDs for annotation |
| POST | `/videos/{id}/map_players` | Map detections → pitch coords |
| GET | `/videos/{id}/homographies/anchor-quality` | Per-keypoint reprojection quality |
| POST | `/videos/{id}/interpolate` | Interpolate + smooth trajectories |
| GET | `/videos/{id}/players` | All player positions |

**v3 endpoint parameters:**

| Param | Default | Range | Meaning |
|-------|---------|-------|---------|
| `num_samples_per_line` | 10 | 2–50 | Points sampled per line annotation |
| `ransac_iterations` | 2000 | 100–10000 | RANSAC trials for initial H |
| `ransac_threshold` | 5.0 | 1.0–50.0 | RANSAC inlier threshold (canvas px) |
| `keypoint_weight` | 20.0 | 1.0–100.0 | Weight ratio keypoints vs line samples |

**Interpolation parameters:**

| Param | Default | Meaning |
|-------|---------|---------|
| `start_frame` | 0 | First frame of output range |
| `end_frame` | 100 | Last frame (inclusive) |
| `sg_long_window` | 15 | SG window for tracks >20 frames |
| `sg_mid_window` | 11 | SG window for tracks 10–20 frames |
| `max_vel_px` | 4.0 | Max px/frame (0 = disabled) |

---

## 8. Key Libraries & Dependencies

### Backend

| Library | Version | Role |
|---------|---------|------|
| `fastapi` | ≥0.100 | HTTP framework, async endpoints |
| `uvicorn` | any | ASGI server |
| `pydantic` | v2 | Request/response validation |
| `opencv-python` | ≥4.8 | Frame extraction, warpPerspective, LK flow, goodFeaturesToTrack |
| `numpy` | ≥1.24 | All matrix operations, SVD |
| `scipy` | any | `savgol_filter` (SG smoothing) |
| `httpx` | any | HTTP client for Modal calls |
| `modal` | any | Serverless GPU platform SDK |
| `ultralytics` | ≥8.0 | YOLOv8 + BotSort (local inference only) |
| `torch` | ≥2.0 | Required by ultralytics (local inference only) |

### Frontend

| Library | Role |
|---------|------|
| `next` | React framework (pages router) |
| `react`, `react-dom` | UI rendering |
| TypeScript | Type safety |

No external UI component libraries — all canvas drawing is hand-coded using the Canvas 2D API.

### Algorithms Used (not library-specific)

| Algorithm | Where | Library Call |
|-----------|-------|-------------|
| RANSAC homography | `homography.py` | `cv2.findHomography(..., cv2.RANSAC)` |
| Hartley normalisation | `homography.py` | Custom (`_hartley_normalize`) |
| Weighted DLT + SVD | `homography.py` | `np.linalg.svd` |
| Lucas-Kanade optical flow | `constrained_homography.py` | `cv2.calcOpticalFlowPyrLK` |
| Shi-Tomasi corners | `constrained_homography.py` | `cv2.goodFeaturesToTrack` |
| Savitzky-Golay filter | `constrained_homography.py`, `trajectories.py` | `scipy.signal.savgol_filter` |
| Linear interpolation | `trajectories.py` | `np.interp` |
| Perspective warping | `rendering.py` | `cv2.warpPerspective` |

---

## 9. Configuration & Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `GPU_PROVIDER` | `"local"` | `"modal"` for remote GPU, `"local"` for CPU fallback |
| `GPU_ENDPOINT_URL` | — | Full URL of the deployed Modal endpoint (required when `GPU_PROVIDER=modal`) |
| `GPU_API_KEY` | — | Optional auth key (not currently used by Modal public endpoints) |
| `MAX_VIDEO_SIZE_MB` | `500` | Upload size limit |
| `ALLOWED_ORIGINS` | `"*"` | Comma-separated CORS origin list |
| `DATA_DIR` | `"data"` | Root directory for all persisted files |
| `YOLO_MODEL_PATH` | `"models/v8s_960_v9.pt"` | Path to YOLO weights for local inference |

**Frontend:**
| Variable | Default | Description |
|----------|---------|-------------|
| `NEXT_PUBLIC_API_URL` | `"http://localhost:8000"` | Backend base URL (set at build time) |

---

## 10. Frontend Architecture

### State Management

All shared state lives in `pages/index.tsx` (`Home` component). Child components are prop-driven and communicate back via callbacks — no global state library (Redux, Zustand, etc.).

**Stale-step invalidation:** when annotations change after running the pipeline, `staleSteps` (a `Set<string>`) marks downstream steps as outdated with "STALE" badge UI. This prevents the user from accidentally using results computed from old annotations. `stepDoneRef` (a `useRef`) tracks which steps have results without triggering re-renders.

### Component Communication

```
Home (index.tsx)
 │  props: videoMetadata, anchorFrames, step results
 ▼
AnchorFrameAnnotator ──► onAnchorFramesChange ──► Home
PipelineSteps        ──► onStepXComplete      ──► Home
ResultsViewer        ──► onFrameChange        ──► Home
DebugLog             ──► onClear              ──► Home
```

### Key Frontend Patterns

**Annotation canvas coordinate system:**
- Canvas buffer size: `min(naturalWidth, 1600) × proportional height` (higher than 1600px is unnecessary; lower than 1000px causes imprecision).
- CSS `outline` not `border` on the canvas element (border shifts `getBoundingClientRect`, causing systematic click offset).
- Zoom via CSS `style.width = canvas.width * zoom` — formula `(clientX - rect.left) * naturalWidth / rect.width` naturally handles any zoom level.

**Race condition prevention in frame loading:**
`loadingFrameIdxRef.current` is set to `frameIdx` before each load. The `onload` callback checks if the ref still matches before updating state — if the user navigates away before a slow load completes, the stale image is discarded.

**Cache busting:**
After step B, `stepBVersion` state is incremented. Warped-frame thumbnail `<img>` elements include `?v={stepBVersion}` in their `src`, forcing the browser to re-fetch after a new homography computation.

**RAF playback:**
`video.play().then(() => requestAnimationFrame(onPlaybackFrame))` — RAF only starts once play() confirms the video is playing. Prevents the pitch canvas updating while the video remains paused.

**Short-track filter:**
In `drawPitch`, tracks with fewer than 30 position entries across the entire dataset are hidden. This suppresses YOLO false positives (brief detections of crowd members or shadows) without removing them from the underlying data.

---

## 11. Pitch Geometry Reference

A GAA pitch is 85 m wide (x) × 140 m long (y). All features are symmetric across both the x=42.5m centreline and the y=70m halfway line.

```
y=0   ─────────────── Top endline ────────────────
      ┌──────────────────────────────────────────┐
      │   Goal area (4.5m)                       │
      │  ┌────┐                        ┌────┐    │  y=4.5m
      │  │    │                        │    │    │
      │  └────┘                        └────┘    │
      │                                          │
      │──────────────────────────────────────────│  y=13m  (13m line)
      │                                          │
      │──────────────────────────────────────────│  y=20m  (20m line)
      │                                          │
      │         ╭──────────────╮                 │  (20m semicircle, r=13m)
      │                                          │
      │──────────────────────────────────────────│  y=45m  (45m line)
      │                                          │
      │──────────────────────────────────────────│  y=65m  (65m line)
      │──────────────────────────────────────────│  y=70m  (halfway)
      │──────────────────────────────────────────│  y=75m
      │──────────────────────────────────────────│  y=95m
      │                                          │
      │         ╰──────────────╯                 │  (20m semicircle, r=13m)
      │──────────────────────────────────────────│  y=120m
      │                                          │
      │──────────────────────────────────────────│  y=127m
      │  ┌────┐                        ┌────┐    │
      │  │    │                        │    │    │  y=135.5m
      │  └────┘                        └────┘    │
      │   Goal area (4.5m)                       │
      └──────────────────────────────────────────┘
y=140 ─────────────── Bottom endline ─────────────
      x=0                             x=85m
      │←─33m─→│←──────19m──────→│←─33m→│
               x=33m             x=52m   (13m box sides)
               x=29.5m         x=55.5m  (small arc sides)
               x=35.5m         x=49.5m  (goalie box)
               x=39.25m        x=45.75m (goal posts)
```

---

## 12. Known Limitations & TODOs

### Algorithmic Limitations

**Homography assumes a flat pitch.**
The pipeline uses a pure perspective homography (8 DOF). Real pitches have slight convexity (~0.3m crown). This introduces a small but systematic error for players near the sidelines at high field positions. A radial distortion term was explored (k1=8e-8) but not retained in the final system.

**No camera model.**
The homography is computed from annotations without any intrinsic camera calibration. Zoom changes between anchor frames will cause the propagated Hs to be incorrect for those frames. The optical flow drift correction partially compensates, but a full camera model (focal length + principal point) would improve accuracy.

**Optical flow fails in low-texture regions.**
If the camera pans to a region of uniform grass with few Shi-Tomasi corners, `_lk_inter_frame_H` returns `None` for that pair. The previous frame's H is reused. In practice this is rare but can cause position discontinuities during fast camera movements.

**BotSort track fragmentation.**
When a player is occluded for many frames, BotSort may assign a new `track_id` when they reappear. The trajectory interpolation only fills within each track's [first, last] detection range — it cannot bridge fragmented tracks. Post-hoc track merging is not implemented.

**Fixed anchor interval.**
The user picks one interval (e.g. every 1 second) for all anchor frames. Dense regions (e.g. fast camera movement) may benefit from more frequent anchors. Sparse regions waste annotation effort.

**Referee track removal is all-or-nothing.**
If a player is ever misclassified as a referee by the YOLO model across all their frames, their entire track is dropped. This is a conservative design choice but may occasionally remove real players.

### Implementation TODOs

- **Backend cleanup (active plan):** Replace remaining magic numbers in `homography.py`, `constrained_homography.py`, `trajectories.py`, `line_constraints.py` with named constants. See `CLAUDE.md` for the full phased plan.
- **No test coverage for `constrained_homography.py`:** The optical flow propagation module has no unit tests. The 70-test suite covers schemas, endpoints, and homography computation but not the LK flow logic.
- **`computeHomographiesV2` in `api.ts`:** A v2 endpoint function remains in `api.ts` but is never called by `PipelineSteps.tsx` (which calls the v3 endpoint directly via `apiFetch`). It could be removed.
- **`PitchAnnotation` in `schemas.py`:** A legacy model that is never used by any active endpoint — exists as a type reference only.
- **No video length limit enforcement beyond file size:** A 500MB file at 25fps could be very long. There is no per-frame limit, but Modal has a 600-second timeout.
- **Homographies not versioned:** If the user re-annotates and re-runs step B, the new homographies overwrite the old ones with no rollback. Player positions from step C (computed against the old Hs) are now stale — the UI shows a STALE badge but doesn't prevent using stale positions.
- **`data/` directory not cleaned up:** Old video files and their associated JSON files accumulate indefinitely. No garbage collection or TTL is implemented.

---

## 13. Development History — What Was Tried, What Worked, What Didn't

### Homography Computation Iterations

**v1 — Keypoints only (RANSAC)**
The initial approach. User annotates visible pitch feature intersections (corner flags, goal post bases, yard-line/sideline intersections). `cv2.findHomography(RANSAC)` computes H.

*What didn't work:* In midfield, very few identifiable point intersections are visible. The homography extrapolated poorly to the centre of the pitch. Player positions near the centre circle were up to 15m off their actual positions.

**v2 — Keypoints + line constraints (ORB propagation)**
Added `LineAnnotation` and a weighted DLT system. User clicks two points on a visible yard line; the system samples N points along the segment and adds them as 1D constraints.

Propagation between anchor frames used ORB feature matching.

*What didn't work:*
- ORB matched moving players as features, causing propagated Hs to drift toward the players.
- ORB matching failed entirely in low-texture regions (uniform grass).
- An early bug had the chaining direction backwards: `H[t] = H[t-1] @ OF_H` instead of `H[t] = H[t-1] @ inv(OF_H)`. This caused systematic drift away from the correct H as more frames were chained.
- Without Hartley normalisation, the DLT SVD was numerically unstable for certain annotation configurations, producing extreme H matrices.

**v3 — Hartley-normalised DLT + Lucas-Kanade propagation (current)**
- Replaced ORB with LK optical flow (forward-backward filter removes moving players).
- Added Hartley normalisation (fixed SVD instability).
- Corrected chaining direction.
- Added linear drift correction between anchor pairs.
- Added Savitzky-Golay smoothing of H elements.
- Added sanity-check fallback to RANSAC-only H.

*What works:* Reprojection errors on anchor frames are consistently < 15px (good quality) for well-annotated frames. Propagated frames maintain reasonable accuracy over ~30-frame segments.

### Annotation UI Iterations

**5px filled circle → crosshair marker**
Initial markers were 5px filled circles. The circle extends 5px in all directions from the click point, so the visual centre of the circle appeared shifted relative to the intended pixel. Replaced with a precision crosshair (2px circle + 7px arms) that centres precisely on the clicked pixel.

**Border → outline (click offset bug)**
Canvas had CSS `border: 2px solid`. `getBoundingClientRect()` returns the border-box, shifting `rect.left/top` by 2px. This caused a ~4–8px systematic error in image-space click coordinates. Fixed by replacing `border` with CSS `outline` (drawn outside the layout box, does not affect `getBoundingClientRect`).

**Canvas buffer 1000px → 1600px**
The annotation canvas buffer was initially capped at 1000px wide. For 1920px-wide video, this introduced ~1px quantisation error per annotation. Increased to 1600px for better precision.

**Smoothing applied only to interpolated gaps → all frames**
Originally, the SG filter was applied to interpolated frames but detected frames were written back with their raw (unsmoothed) coordinates. This caused visible jitter at anchor frames where the trajectory jumped between the smoothed interpolated sequence and the raw detected value. Fixed by applying smoothed coordinates to all frames.

### Radial Distortion (Explored, Not Retained)

An experimental `k1` parameter for radial lens distortion was added to the frontend. The idea was to apply a radial correction to player positions after homography mapping. The parameter was exposed as a UI slider.

*Outcome:* The distortion correction did not meaningfully improve tracking accuracy on the available test footage, and the UI element added clutter. The feature was removed.

### PTZ Decomposition (Explored, Abandoned)

`pipeline_testing_and_research/diag_ptz_decompose.py` explored decomposing each homography into Pan/Tilt/Zoom parameters to constrain the propagation model (e.g. enforce that zoom doesn't change between closely-spaced frames). This approach was not pursued because:
- The decomposition requires known camera intrinsics.
- The direct optical flow approach was already robust enough.

---

## 14. Getting Started for a New Engineer

### Running Locally

**Backend:**
```bash
cd interactive_analytics_system_backend
pip install -r requirements.txt
uvicorn app:app --reload --port 8000
```

Default `GPU_PROVIDER=local` will run YOLO on CPU (slow). For GPU inference, set `GPU_PROVIDER=modal` and `GPU_ENDPOINT_URL` after deploying the Modal service.

**Frontend:**
```bash
cd interactive_analytics_system_frontend
npm install
NEXT_PUBLIC_API_URL=http://localhost:8000 npm run dev
```

### Running Tests
```bash
cd interactive_analytics_system_backend
pytest tests/ -v
```
70 tests pass. 1 pre-existing failure: `test_validate_tilted_line_fails` in `test_line_constraints.py` (unrelated to core pipeline).

### Deploying the Modal GPU Service
```bash
pip install modal
modal token new
modal volume put yolo-model-cache path/to/v8s_960_v9.pt /v8s_960_v9.pt
modal deploy gpu_inference/modal_yolo.py
# Copy the printed endpoint URL
export GPU_ENDPOINT_URL="https://..."
export GPU_PROVIDER=modal
```

### Where to Look for What

| Question | Where to look |
|----------|--------------|
| How does the homography algorithm work? | `pipeline/HOMOGRAPHY.md`, `pipeline/homography.py` |
| Why is Hartley normalisation needed? | `pipeline/HOMOGRAPHY.md` §"Why mandatory", docstring in `_hartley_normalize` |
| How does per-frame propagation work? | `pipeline/OPTICAL_FLOW.md`, `pipeline/constrained_homography.py` |
| How are players mapped to the pitch? | `pipeline/PLAYER_TRACKING.md`, `pipeline/map_players.py` |
| How does trajectory smoothing work? | `pipeline/PLAYER_TRACKING.md`, `pipeline/trajectories.py` |
| What are the pitch geometry constants? | `pipeline/CONFIG.md`, `pipeline/gaa_pitch_config.py` |
| How does the annotation UI work? | `components/ANCHOR_FRAME_ANNOTATOR.md` |
| Why does the canvas use outline not border? | `components/ANCHOR_FRAME_ANNOTATOR.md` §"canvasEventToImageCoords" |
| How does GPU inference work? | `gpu_inference/OVERVIEW.md` |
| What endpoints exist and what do they return? | `APP_ENDPOINTS.md`, this document §7 |
| How is state managed in the frontend? | `OVERVIEW.md` (frontend), `pages/INDEX.md` |

### Common Gotchas

1. **Coordinate system confusion.** Always check whether a position is in image pixels, pitch-canvas pixels, or pitch meters. The debug coordinate table in ResultsViewer (visible in the UI) shows all three for the current frame.

2. **H matrix direction.** All Hs map image → canvas, not canvas → image. If you need the inverse mapping, use `np.linalg.inv(H)`.

3. **JSON string keys.** Frame indices are stored as string keys in JSON (`"0"`, `"25"`, etc.) but as integer keys in Python dicts. `_deserialize_H` handles the conversion; don't assume integer keys when loading from disk.

4. **Class name strings.** The YOLO model class names contain typos (`"Refree-lablers"`, `"GAA-player-lablers"`) that must match exactly. Do not "fix" these.

5. **Smoothing applied to all frames.** The SG filter is intentionally applied to detected frames as well as interpolated gaps. Reverting this causes jitter at anchor frames.

6. **Re-pinning after smoothing.** After applying SG smoothing to H elements, anchor frames must be re-pinned exactly (`per_frame_H[A] = anchor_homographies[A]`). If you skip this, anchor frames are slightly perturbed by the smoothing polynomial.

7. **`outline` vs `border` on annotation canvas.** Any CSS property that affects the border-box width/height of the annotation canvas (including `padding` and `border`) will corrupt click coordinate conversion. Only `outline` is safe.
