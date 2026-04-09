# GAA Video Analysis — Complete System Walkthrough

> **Audience:** A junior or intern engineer who has just cloned the repo and wants to understand exactly what happens, end to end, from the moment a user presses Upload to the moment they see KPI results. This document follows a single user session in real time, naming every file, every function, and every piece of data that flows through the system.

---

## Table of Contents

1. [Big-Picture Architecture](#1-big-picture-architecture)
2. [Repository Map — Where Everything Lives](#2-repository-map--where-everything-lives)
3. [Step 0 — The App Boots](#step-0--the-app-boots)
4. [Step 1 — User Uploads a Video](#step-1--user-uploads-a-video)
5. [Step 2 — The UI Asks for Anchor Frames](#step-2--the-ui-asks-for-anchor-frames)
6. [Step 3 — User Annotates Anchor Frames](#step-3--user-annotates-anchor-frames)
7. [Step 4A — Run Tracking (YOLO + BotSort)](#step-4a--run-tracking-yolo--botsort)
8. [Step 4B — Compute Homographies](#step-4b--compute-homographies)
9. [Step 4C — Map Players to Pitch](#step-4c--map-players-to-pitch)
10. [Step 4D — Interpolate Trajectories](#step-4d--interpolate-trajectories)
11. [Step 5 — Results Viewer Loads](#step-5--results-viewer-loads)
12. [Step 6 — Classify Teams (Optional)](#step-6--classify-teams-optional)
13. [Step 7 — Compute KPIs (Optional)](#step-7--compute-kpis-optional)
14. [The Three Coordinate Systems](#the-three-coordinate-systems)
15. [The In-Memory Store and Disk Layout](#the-in-memory-store-and-disk-layout)
16. [Common Bugs and Where to Look](#common-bugs-and-where-to-look)

---

## 1. Big-Picture Architecture

The system has two processes: a **Next.js frontend** (browser) and a **FastAPI backend** (Python server). A third service — a **Modal GPU worker** — handles the heavy YOLO inference in the cloud.

```
┌──────────────────────────────────────────────────────────────────────────────┐
│  Browser  (Next.js / React / TypeScript)                                     │
│                                                                              │
│  pages/index.tsx          ← root of all state: videoMetadata, positions,     │
│     │                       annotations, team classifications, KPI summary   │
│     ├── VideoUploader.tsx ← drag-and-drop file picker                        │
│     ├── AnchorFrameAnnotator.tsx ← frame + pitch diagram side-by-side canvas │
│     ├── PipelineSteps.tsx ← buttons that fire backend API calls              │
│     └── ResultsViewer.tsx ← video + 2D pitch playback, KPI panels            │
│                                                                              │
│  lib/api.ts               ← every fetch() call lives here                    │
│  lib/pitchConfig.ts       ← pitch geometry constants (mirrors backend)       │
│  lib/pitch.ts             ← canvas drawing functions                         │
│  utils/kpiUtils.ts        ← KPI zone analysis helpers                        │
└──────────────────────────────┬───────────────────────────────────────────────┘
                               │  HTTP (JSON / JPEG)
                               ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  FastAPI Backend  (Python / OpenCV / NumPy / SciPy)                          │
│                                                                              │
│  app.py                   ← wire up routes, CORS, startup lifecycle          │
│  store.py                 ← in-memory state (VideoStore singleton)           │
│                                                                              │
│  routes/                                                                     │
│    videos.py              ← upload, frame serving, warped view               │
│    detection.py           ← trigger tracking, serve detections               │
│    homography.py          ← anchor H computation + quality check             │
│    mapping.py             ← map players, interpolate, fetch positions        │
│    classification.py      ← team colour classification + overrides           │
│    kpi.py                 ← compute KPI summary                              │
│    deps.py                ← shared get_video_or_404 helper                   │
│                                                                              │
│  pipeline/                                                                   │
│    config.py              ← OUT_W, OUT_H (re-exports from gaa_pitch_config)  │
│    gaa_pitch_config.py    ← pitch geometry: vertices, lines, sidelines       │
│    schemas.py             ← Pydantic models (Detection, PitchPoint, etc.)    │
│    video.py               ← OpenCV wrappers for metadata + frame extraction  │
│    rendering.py           ← draw reference lines onto warped frames          │
│    homography.py          ← v3 DLT + Hartley normalisation algorithm         │
│    line_constraints.py    ← sample_points_on_line, re-exports line dicts     │
│    constrained_homography.py ← LK optical flow propagation                  │
│    map_players.py         ← filter detections, apply H to bbox bottom-centre │
│    trajectories.py        ← linear interp → SG smooth → velocity clamp       │
│    team_classifier.py     ← jersey HSV analysis per track                    │
│    kpi.py                 ← distance, convex hull, zone balance, centroids   │
│    persistence.py         ← all disk I/O in one place                        │
└──────────────────────────────┬───────────────────────────────────────────────┘
                               │  HTTP (base64 video → JSON detections)
                               ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  Modal GPU Worker  (Serverless, T4 GPU)                                      │
│  gpu_inference/modal_yolo.py  ← YOLO v8-small + BotSort tracking            │
│  gpu_inference/__init__.py    ← HTTP client + Detection conversion           │
└──────────────────────────────────────────────────────────────────────────────┘
```

The frontend and backend talk over REST. The backend stores state partly in memory (`store.py`) and partly on disk. The Modal GPU worker is stateless — it receives a video, runs inference, and returns JSON.

---

## 2. Repository Map — Where Everything Lives

```
GAA-Video-Analysis/
├── interactive_analytics_system_backend/    ← Python FastAPI server
│   ├── app.py                              ← FastAPI app object, router wiring, startup
│   ├── store.py                            ← VideoStore singleton (in-memory state)
│   ├── main.py                             ← uvicorn entry point for Render deployment
│   ├── pipeline/
│   │   ├── gaa_pitch_config.py             ← ALL pitch geometry constants live here
│   │   ├── config.py                       ← OUT_W, OUT_H (thin re-export), YOLO path
│   │   ├── schemas.py                      ← Pydantic models for every data type
│   │   ├── video.py                        ← get_video_metadata(), extract_frame()
│   │   ├── rendering.py                    ← draw_reference_lines() for warped JPEG
│   │   ├── homography.py                   ← compute_homographies_with_lines_v3()
│   │   ├── line_constraints.py             ← sample_points_on_line(), re-exports line dicts
│   │   ├── constrained_homography.py       ← build_optical_flow_per_frame_H()
│   │   ├── map_players.py                  ← filter_detections_for_mapping(), map_players_to_pitch()
│   │   ├── trajectories.py                 ← interpolate_trajectories()
│   │   ├── team_classifier.py              ← classify_tracks(), override_classification()
│   │   ├── kpi.py                          ← compute_clip_summary()
│   │   └── persistence.py                  ← all save_*/load_* helpers, restore_videos_from_disk()
│   ├── routes/
│   │   ├── deps.py                         ← get_video_or_404() shared dependency
│   │   ├── videos.py                       ← POST /videos, GET /frame, GET /warped
│   │   ├── detection.py                    ← POST /track, GET /detections
│   │   ├── homography.py                   ← POST /homographies/v3, GET /anchor-quality
│   │   ├── mapping.py                      ← POST /map_players, POST /interpolate, GET /players
│   │   ├── classification.py               ← POST/GET/PATCH /classify-teams
│   │   └── kpi.py                          ← POST /compute-kpis
│   ├── gpu_inference/
│   │   ├── modal_yolo.py                   ← Modal serverless GPU service
│   │   └── __init__.py                     ← GPUInferenceClient (HTTP client)
│   └── tests/                              ← pytest suite
│
├── Interactive_analytics_system_frontend/   ← Next.js / TypeScript
│   ├── pages/index.tsx                     ← root page, all cross-step state lives here
│   ├── components/
│   │   ├── VideoUploader.tsx               ← file drag-and-drop
│   │   ├── AnchorFrameAnnotator.tsx        ← annotation canvas UI (~670 lines)
│   │   ├── PipelineSteps.tsx               ← step runner buttons + quality display
│   │   ├── ResultsViewer.tsx               ← video + pitch playback
│   │   ├── TeamClassificationPanel.tsx     ← jersey swatches + override selects
│   │   ├── ClipSummaryCard.tsx             ← plain-English KPI card
│   │   └── KpiPanel.tsx                    ← detailed KPI table
│   ├── lib/
│   │   ├── api.ts                          ← ALL fetch() calls — touch only this for API changes
│   │   ├── pitchConfig.ts                  ← single source of truth for pitch geometry (TS)
│   │   ├── pitch.ts                        ← canvas drawing: drawPitchDiagram(), drawPitch()
│   │   └── constants.ts                    ← re-exports pitchConfig.ts (backward compat shim)
│   ├── utils/
│   │   ├── kpiUtils.ts                     ← computeZoneAnalysis(), computeDepthSentence()
│   │   ├── formatters.ts                   ← reprErrorLabel(), qualityBadge(), etc.
│   │   └── canvasUtils.ts                  ← drawCrosshair()
│   └── types/index.ts                      ← ALL TypeScript interfaces
│
└── pipeline_testing_and_research/          ← DO NOT MODIFY (diagnostic scripts)
```

> **Golden rule:** if you want to add a new API call, the change touches exactly two files — `routes/<relevant>.py` on the backend and `lib/api.ts` on the frontend. No other files should need editing for a simple endpoint addition.

---

## Step 0 — The App Boots

### Backend startup

**File:** `app.py` → `lifespan()`

When you run `uvicorn app:app` (or `python main.py`), FastAPI calls the `lifespan` async context manager before accepting any requests:

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    ensure_dirs()                           # create data/videos/, data/tracks/, data/annotations/
    videos = restore_videos_from_disk()     # scan data/videos/*_meta.json
    store.videos.update(videos)             # repopulate in-memory store
    yield                                   # server accepts requests from here
```

`ensure_dirs()` and `restore_videos_from_disk()` live in `pipeline/persistence.py`. This is the only disk I/O at startup. All other caches (detections, homographies, player positions) start empty and are loaded lazily when a request needs them.

The six FastAPI routers are included in `app.py`:
```python
app.include_router(videos_router)        # routes/videos.py
app.include_router(detection_router)     # routes/detection.py
app.include_router(homography_router)    # routes/homography.py
app.include_router(mapping_router)       # routes/mapping.py
app.include_router(classification_router)# routes/classification.py
app.include_router(kpi_router)           # routes/kpi.py
```

### Frontend startup

**File:** `pages/index.tsx`

The Next.js root page initialises all shared state in one place with `useState`:

```typescript
const [videoMetadata, setVideoMetadata] = useState<VideoMetadata | null>(null)
const [anchorFrames, setAnchorFrames] = useState<number[]>([])
const [annotations, setAnnotations] = useState<Record<number, AnnotationState>>({})
const [playerPositions, setPlayerPositions] = useState<PlayerPosition[]>([])
const [teamClassifications, setTeamClassifications] = useState<TeamClassifications>({})
const [kpiSummary, setKpiSummary] = useState<KpiSummary | null>(null)
// ... etc
```

This is intentional: the page is the "controller". Every child component receives the state it needs as props. When a child needs to update shared state (e.g. new annotations), it calls a setter passed down from the page.

---

## Step 1 — User Uploads a Video

### 1a. Frontend

**File:** `components/VideoUploader.tsx`

The user drags a `.mp4` file onto the uploader or uses the file input. The component calls `onFileSelected(file)`, which is a prop callback defined in `pages/index.tsx`.

`index.tsx` then calls:

```typescript
// lib/api.ts
const metadata = await uploadVideo(file)   // POST /videos (multipart form data)
setVideoMetadata(metadata)
setVideoFile(file)    // stored as a browser File object — used later for the <video> element
```

### 1b. Backend

**File:** `routes/videos.py` → `upload_video()`

```python
@router.post("/videos", response_model=VideoCreateResponse)
async def upload_video(file: UploadFile = File(...)):
    content = await file.read()
    validate_video_upload(file, content)      # size ≤ 500MB, .mp4 extension, correct MIME

    video_id = str(uuid.uuid4())              # random UUID — this is the stable key for everything
    video_path = save_video_file(video_id, content)   # write to data/videos/{uuid}.mp4

    metadata = get_video_metadata(str(video_path))    # OpenCV: fps, num_frames, width, height
    store.videos[video_id] = video_meta               # in-memory
    save_video_meta(video_id, video_meta)             # persist to data/videos/{uuid}_meta.json

    return VideoCreateResponse(video_id=video_id, ...)
```

`get_video_metadata()` in `pipeline/video.py` opens the file with `cv2.VideoCapture`, reads `CAP_PROP_FPS`, `CAP_PROP_FRAME_COUNT`, `CAP_PROP_FRAME_WIDTH`, `CAP_PROP_FRAME_HEIGHT`, closes it, and returns a dict. Simple — but everything downstream depends on `num_frames` and `fps` being correct.

`save_video_meta()` and `save_video_file()` live in `pipeline/persistence.py`. All file I/O in the system goes through that module — nowhere else does an `open()` call appear.

**What the frontend gets back:** `{ video_id, fps, num_frames, width, height, duration_seconds }` — stored as `videoMetadata`.

---

## Step 2 — The UI Asks for Anchor Frames

**File:** `pages/index.tsx`, `components/AnchorFrameAnnotator.tsx`

After uploading, the user configures how many anchor frames to annotate. An anchor frame is a frame where the user will manually mark pitch landmarks — these are the frames on which homographies are computed. More anchors = better coverage, but more user effort.

The UI typically spaces anchors evenly across the video:
```typescript
// Every N frames (e.g. every second = every fps frames)
const anchors = Array.from({ length: numAnchors }, (_, i) =>
  Math.round(i * (numFrames - 1) / (numAnchors - 1))
)
setAnchorFrames(anchors)
```

`AnchorFrameAnnotator` then loads each anchor frame from the backend on demand:
```typescript
// lib/api.ts → GET /videos/{id}/frame/{idx}
const imgUrl = `${API_URL}/videos/${videoId}/frame/${frameIdx}`
```

The backend serves these via `routes/videos.py` → `get_frame()`, which calls `pipeline/video.py` → `extract_frame()` — this uses `cv2.VideoCapture.set(CAP_PROP_POS_FRAMES, idx)` to seek directly to the frame and returns JPEG bytes.

---

## Step 3 — User Annotates Anchor Frames

**File:** `components/AnchorFrameAnnotator.tsx`

This is the most complex UI component (~670 lines). The user works on two canvases side-by-side:
- **Left:** the video frame (served as a JPEG from `GET /videos/{id}/frame/{idx}`)
- **Right:** a vector pitch diagram drawn via `lib/pitch.ts` → `drawPitchDiagram()`

### 3a. Point Mode

The user clicks a feature in the video frame (a corner flag, goal post, line intersection). A crosshair appears (`utils/canvasUtils.ts` → `drawCrosshair()`). Then they click the corresponding location in the pitch diagram.

The pitch diagram click is snapped to the nearest known vertex or line segment. Named vertices come from `lib/pitchConfig.ts` → `GAA_PITCH_VERTICES` (e.g. `"corner_tl"`, `"left_13m_box_top"`). Line segment snapping checks proximity to the segments in `PITCH_LINE_SEGMENTS`.

The result is a `PitchPoint`:
```typescript
interface PitchPoint {
  pitch_id: string   // e.g. "corner_tl" or "line_45m_top_x0_y45"
  x_img: number      // image pixel coordinate (original resolution, not display scale)
  y_img: number
}
```

**Critical implementation detail:** `x_img` and `y_img` are in **original image pixels** (0..naturalWidth/Height), not display pixels. The conversion at click time:
```typescript
const x = (e.clientX - rect.left) * img.naturalWidth / rect.width
const y = (e.clientY - rect.top)  * img.naturalHeight / rect.height
```
`rect.width` is the displayed pixel width of the canvas (which changes with zoom). Dividing by it and multiplying by `naturalWidth` gives the original-resolution coordinate. This formula is zoom-independent — it works the same at 1×, 2×, 3×, 4× zoom.

**The outline vs border fix:** The canvas uses `outline: 2px` rather than `border: 2px`. `getBoundingClientRect()` returns the border-box — a 2px border would push `rect.left` inward by 2px, causing a systematic ~8px error in image space. `outline` is drawn outside the layout box and does not affect `getBoundingClientRect()`.

### 3b. Line Mode

The user selects a pitch line from a dropdown (populated from `AVAILABLE_LINES` in `lib/pitchConfig.ts`), then clicks two points that lie on that line in the video frame. This creates a `LineAnnotation`:
```typescript
interface LineAnnotation {
  line_id: string   // e.g. "45m_top"
  u1: number; v1: number   // first point, image pixels
  u2: number; v2: number   // second point, image pixels
}
```

Lines are useful in midfield, where there are no named intersections visible (no corner flags, no box corners). The backend will sample 10 points along the `(u1,v1)→(u2,v2)` segment and use them as 1D constraints (the Y-coordinate is known from the line's pitch position).

### 3c. Persistence

Annotations are saved to `localStorage` on every change:
```typescript
localStorage.setItem(`gaa_annotations_${videoFilename}`, JSON.stringify(annotations))
```

This means re-uploading the same video restores the annotations automatically — useful during development. The backend only receives annotations when the user clicks "Run Pipeline".

---

## Step 4A — Run Tracking (YOLO + BotSort)

### Frontend trigger

**File:** `components/PipelineSteps.tsx` → `runStepA()`

```typescript
const data = await trackVideo(videoMetadata.video_id)  // POST /videos/{id}/track
```

### Backend

**File:** `routes/detection.py` → `track_video()`

```python
@router.post("/videos/{video_id}/track")
async def track_video(video_id: str):
    detections = load_detections(video_id)      # check disk first (idempotent)
    if detections is None:
        from gpu_inference import get_gpu_client
        client = get_gpu_client()
        detections = client.track_video(video_path)   # remote GPU call
        store.detections_cache[video_id] = detections
        save_detections(video_id, detections)
```

The `load_detections` check makes this endpoint **idempotent** — re-running does nothing if tracks already exist. This is intentional: tracking takes ~2–5 minutes and you never want to redo it accidentally.

### GPU Inference (Modal)

**Files:** `gpu_inference/__init__.py`, `gpu_inference/modal_yolo.py`

`get_gpu_client()` returns a `GPUInferenceClient` singleton. `client.track_video(video_path)` reads the video file, base64-encodes it, and POSTs it to the Modal HTTP endpoint:

```python
# gpu_inference/__init__.py
def _track_modal(self, video_path: str) -> List[Detection]:
    with open(video_path, "rb") as f:
        video_b64 = base64.b64encode(f.read()).decode()
    response = self.http.post(self.endpoint_url, json={"video_b64": video_b64})
    return [Detection(**d) for d in response.json()]
```

On the Modal side (`gpu_inference/modal_yolo.py`), a containerised `YOLOTracker` class:
1. Decodes the base64 video to a temp file.
2. Calls `model.track(source=tmp_path, stream=True, conf=0.35, imgsz=960, tracker="botsort.yaml")`.
3. For each frame result, iterates over boxes, extracts `track_id`, class name, bbox, confidence.
4. Returns a list of dicts.

**Output:** A `List[Detection]` — one entry per player per frame:
```python
class Detection(BaseModel):
    frame_idx: int
    track_id: int       # stable across frames for the same player
    x1: float; y1: float; x2: float; y2: float   # bounding box in image pixels
    confidence: float
    class_name: str     # "GAA-player-lablers", "Ball-labelers", or "Refree-lablers"
```

Saved to `data/tracks/{uuid}.json`.

---

## Step 4B — Compute Homographies

This is the most mathematically dense step. Read carefully.

### Frontend trigger

**File:** `components/PipelineSteps.tsx` → `runStepB()`

```typescript
const annotations: AnchorFrameAnnotation[] = Object.entries(validAnnotations)
  .map(([frameIdx, ann]) => ({
    frame_idx: parseInt(frameIdx),
    points: ann.keypoints,
    lines: ann.lines,
  }))
const result = await computeHomographies(videoMetadata.video_id, annotations)
// lib/api.ts → POST /videos/{id}/homographies/v3
```

### What is a homography and why do we need it?

A **homography** is a 3×3 matrix H that maps one set of 2D points to another when the relationship between them is a projective transformation (as when you photograph a flat surface from an angle). For a stationary camera looking at a flat pitch, a single H would suffice. But a PTZ camera pans, tilts, and zooms — a different H is needed for every frame.

Given a point `(x_img, y_img)` in camera pixels, the homography maps it to a point `(x_canvas, y_canvas)` in our fixed pitch canvas (850×1400 px):

```python
p = H @ [x_img, y_img, 1.0]
p /= p[2]          # perspective division (homogeneous coordinates)
x_canvas, y_canvas = p[0], p[1]
```

### Backend: Anchor H computation

**File:** `routes/homography.py` → `compute_homographies_v3()` → `pipeline/homography.py` → `compute_homographies_with_lines_v3()`

For each anchor frame with ≥4 keypoints:

#### Step B1 — RANSAC H₀ (keypoints only)

```python
H0, _ = cv2.findHomography(
    pts_image.astype(np.float32),    # Nx2 in image pixels
    pts_canvas.astype(np.float32),   # Nx2 in canvas pixels (from GAA_PITCH_VERTICES lookup)
    cv2.RANSAC, 5.0, maxIters=2000
)
```

`pts_canvas` comes from `_meters_to_canvas_pixels()`, which looks up the vertex position in meters from `GAA_PITCH_VERTICES` and multiplies by 10 (px/m). RANSAC handles misannotated keypoints by trying random subsets.

If there are no line annotations, H₀ is used directly. Otherwise, we proceed to refine it.

#### Step B2 — Hartley Normalisation

Before building the full DLT system, both point sets are normalised so their centroid is the origin and mean distance from origin is √2. This is mathematically mandatory:

Without normalisation, the DLT matrix entries are products like `x_img × x_canvas` ≈ 1920 × 1400 ≈ 2.7M. These giant numbers make the SVD numerically unstable — the smallest singular vector (which encodes H) gets contaminated by floating-point error, producing a wildly wrong result.

With normalisation (both sets scaled to ≈(−1.4, 1.4) range), the SVD is well-conditioned.

```python
pts_image_n,  T_img    = _hartley_normalize(pts_image)    # pipeline/homography.py
pts_canvas_n, T_canvas = _hartley_normalize(pts_canvas)
```

#### Step B3 — Weighted DLT system

Each correspondence adds rows to the matrix A. Keypoints add 2 rows (full 2D constraint) with weight 20. Line samples add 1 row (1D constraint — only the Y or X direction is known) with weight 1:

```
# Keypoint (u,v) → (x,y) in normalised coords, weight=20:
[u, v, 1, 0, 0, 0, -x·u, -x·v, -x]   ← constrains the x-output
[0, 0, 0, u, v, 1, -y·u, -y·v, -y]   ← constrains the y-output

# Horizontal line sample (u,v) → known y_c, weight=1:
[0, 0, 0, u, v, 1, -y_c·u, -y_c·v, -y_c]   ← constrains only y-output

# Vertical sideline sample (u,v) → known x_c, weight=1:
[u, v, 1, 0, 0, 0, -x_c·u, -x_c·v, -x_c]   ← constrains only x-output
```

Why the 20:1 ratio? With ~4 keypoints (8 weighted rows) vs ~30 line samples (30 rows), effective weight ratio ≈ 160:30 ≈ 5:1. Keypoints dominate. Lines can only correct directions that keypoints don't cover — e.g. lateral skew in midfield where all keypoints are near one end.

#### Step B4 — SVD solve

```python
_, _, Vt = np.linalg.svd(A * w_vec[:, np.newaxis], full_matrices=False)
H_norm = Vt[-1].reshape(3, 3)   # last row of Vt = null vector (min singular value)
```

The null vector of `A` (the vector Hv such that `Av ≈ 0`) is the least-squares solution to the homogeneous system. SVD gives this as the last row of `Vt`.

#### Step B5 — Denormalise

```python
H = np.linalg.inv(T_canvas) @ H_norm @ T_img
H /= H[2, 2]   # normalise so H[2,2] = 1 (standard form)
```

#### Step B6 — Sanity check

Fall back to H₀ if H is degenerate:
- `np.any(np.isnan(H))` — numerically failed
- `np.linalg.cond(H) > 1e8` — near-singular matrix
- Mean reprojection error of H > 2× that of H₀ — lines actively made things worse

Reprojection error = distance (in canvas pixels) between where H maps an image keypoint vs where that keypoint actually is on the pitch canvas. Good anchors have mean error < 10px.

### Backend: Per-frame propagation

**File:** `pipeline/constrained_homography.py` → `build_optical_flow_per_frame_H()`

We have H matrices for maybe 5–10 anchor frames. We need one for every frame (potentially thousands). This is done with Lucas-Kanade optical flow.

**Phase 1 — Compute inter-frame flow matrices**

For each consecutive pair (t, t+1), compute `H_{t→t+1}` using LK optical flow:

1. `cv2.goodFeaturesToTrack` finds stable corners in the grayscale frame. A mask zeros the top 35% (sky/stands — features there move independently of the pitch camera motion).
2. `cv2.calcOpticalFlowPyrLK` tracks those corners from frame t to t+1 (forward), then backward from t+1 to t.
3. Forward-backward filter: discard any point where `|pt_original − pt_tracked_back| > 1px`. This removes moving players — their flow is inconsistent.
4. `cv2.findHomography(RANSAC)` on surviving points gives `H_{t→t+1}`.

**Phase 2 — Chain and drift-correct per segment**

For each segment between anchor frames A and B:

```python
# Chain forward from A
H[t] = H[t-1] @ inv(of_Hs[t-1])   # for t = A+1 .. B
```

`of_Hs[t-1]` maps frame t-1 to frame t. Its inverse maps t to t-1. Composing with `H[t-1]` (which maps frame t-1 to pitch canvas) gives H[t]: frame t to pitch canvas.

Chaining accumulates drift. At anchor B, the chained estimate `H_chain[B]` may not match the trusted `anchor_H[B]`. Drift correction:
```python
H_drift = anchor_H[B] @ inv(H_chain[B])
alpha   = (t - A) / (B - A)         # 0 at A, 1 at B
H[t]    = ((1-alpha)*I + alpha*H_drift) @ H_chain[t]
```

Both anchor frames are re-pinned exactly (overwritten with trusted H).

**Phase 3 — Savitzky-Golay smoothing**

Each of the 9 H-matrix elements is smoothed independently across the segment using a window of up to 21 frames, polynomial order 2. This removes high-frequency jitter from optical flow estimation noise. Anchor frames are re-pinned again after smoothing.

The per-frame H dict is saved to `data/annotations/{id}_v3_homographies.json`.

---

## Step 4C — Map Players to Pitch

### Frontend trigger

**File:** `components/PipelineSteps.tsx` → `runStepC()` (part of the "Run Pipeline" sequence)

```typescript
await mapPlayers(videoMetadata.video_id)   // POST /videos/{id}/map_players
await interpolateTrajectories(videoMetadata.video_id, 0, numFrames - 1)  // POST /videos/{id}/interpolate
const positions = await getPlayerPositions(videoMetadata.video_id)  // GET /videos/{id}/players
setPlayerPositions(positions)
```

### Backend: Filter

**File:** `routes/mapping.py` → `map_players()` → `pipeline/map_players.py` → `filter_detections_for_mapping()`

Before mapping, detections are cleaned:

1. Drop all `class_name == CLASS_BALL` detections.
2. Find all `track_id` values that have **any** `CLASS_REFEREE` detection. Drop **all** detections for those track IDs (whole-track removal handles occasional misclassifications across frames).
3. Drop all tracks with fewer than 25 total raw detections. At 25fps, this is ~1 second of visibility. Shorter tracks are tracker glitches (the tracker briefly assigns a new ID to a detected object that doesn't persist).

### Backend: Map

**File:** `pipeline/map_players.py` → `map_players_to_pitch()`

For each surviving detection, apply the per-frame H to the **bottom-centre of the bounding box**:

```python
x_foot = (det.x1 + det.x2) / 2    # horizontal centre of bbox
y_foot = det.y2                     # bottom edge of bbox

x_canvas, y_canvas = map_pixel_to_pitch(x_foot, y_foot, H[det.frame_idx])
```

Why bottom-centre? A homography maps points on the **ground plane**. The bottom edge of the bounding box is where the player's feet are — the actual contact point with the ground. Using the bbox centre (the player's torso) or top (their head) introduces a systematic upward offset because the camera is elevated.

Output: `List[PlayerPitchPosition]` in **pitch-canvas pixels** (0..850 × 0..1400), stored in `store.player_positions_cache[video_id]`.

---

## Step 4D — Interpolate Trajectories

**File:** `routes/mapping.py` → `interpolate_trajectories_endpoint()` → `pipeline/trajectories.py` → `interpolate_trajectories()`

Player mapping produces **sparse** positions: only frames where the YOLO model detected and BotSort tracked a player. A player is typically visible in ~60–80% of frames, with gaps when they're behind another player or near the edge of the frame.

Interpolation fills those gaps and smooths the result. Per track (group by `track_id`):

**1. Linear interpolation between all detected positions:**
```python
xs = np.interp(all_frames_in_range, detected_frames, detected_xs)
ys = np.interp(all_frames_in_range, detected_frames, detected_ys)
```
No extrapolation — only fills within `[first_detection, last_detection]`.

**2. Canvas clip:** clamp x to [0, 850], y to [0, 1400].

**3. Savitzky-Golay smoothing:**
| Track length | Window |
|---|---|
| > 20 frames | `min(15, n)` (configurable) |
| 10–20 frames | `min(11, n)` |
| < 10 frames | No smoothing |

Window must be odd. Polynomial order 2. Applied to x and y independently. **Important:** smoothing is applied to the full sequence including originally-detected frames (not just the interpolated gaps). Earlier versions only smoothed gaps, causing visible jitter every time a frame with a raw detection was hit.

**4. Max-velocity clamping:**
```python
for i in range(1, n):
    dist = hypot(dx, dy)
    if dist > max_vel_px:              # default 4.0 px/frame = 10 m/s at 25fps
        xs[i] = xs[i-1] + dx * (max_vel_px / dist)
        ys[i] = ys[i-1] + dy * (max_vel_px / dist)
```
Each violating step is scaled back to the speed limit. The correction is local — it doesn't propagate forward (unlike some accumulating clamping approaches that cause drift).

The interpolated positions are merged into `store.player_positions_cache[video_id]` and fetchable via `GET /videos/{id}/players`.

---

## Step 5 — Results Viewer Loads

**File:** `components/ResultsViewer.tsx`, `lib/pitch.ts`

Once positions are available, `index.tsx` switches the UI to the results viewer. This is a side-by-side layout:

- **Left:** HTML `<video>` element using a blob URL created from the uploaded `File` object (never re-downloaded from the server — the `File` is kept in React state).
- **Right:** `<canvas>` showing a 2D bird's-eye pitch with coloured player dots.

### Frame sync

The video and canvas are kept in sync via `currentFrame` state. Two sync mechanisms exist:

1. **Scrubbing:** The user drags the frame slider → `setCurrentFrame(idx)` → the video seeks via `video.currentTime = idx / fps`.
2. **Playback loop:**
   ```typescript
   animFrameRef.current = requestAnimationFrame(onPlaybackFrame)
   // onPlaybackFrame reads video.currentTime, converts to frame index, calls onFrameChange
   ```
   The RAF loop only starts once `video.play()` resolves.

### Canvas drawing

**File:** `lib/pitch.ts` → `drawPitch()`

On every frame change, the pitch canvas is redrawn:

1. **Pitch background + markings** (`drawPitchDiagram()`): green fill, white lines derived from `GAA_PITCH_LINES`, `GAA_PITCH_SIDELINES`, `PITCH_SYMMETRIC_LINE_PAIRS` in `lib/pitchConfig.ts`. All coordinates are in meters; `pitchToCanvas()` converts to the 340×560 display canvas.

2. **Ghost dots:** For tracks visible in the recent past but not this frame, a semi-transparent grey dot at their last known position.

3. **Active player dots:** For each `PlayerPitchPosition` at `currentFrame`:
   - Scale from backend coords (850×1400) to display (340×560): `x_display = x_pitch / PITCH_CANVAS_W * PITCH_DISPLAY_WIDTH`
   - Colour depends on team classification:
     - `'ellistown'` → gold `#FFD700`
     - `'opposition'` → blue `#4488FF`
     - `'referee'` or `'ignore'` → hidden entirely
     - Unclassified → `hsl((track_id × 137.508) % 360, 70%, 50%)` — the golden angle ensures maximally distinct hues for adjacent track IDs

---

## Step 6 — Classify Teams (Optional)

**File:** `components/ResultsViewer.tsx`, `components/TeamClassificationPanel.tsx`

The user clicks "Classify Teams". This is optional — the system works without it, but player dots will be rainbow-coloured instead of team-coloured.

### Frontend

```typescript
// lib/api.ts
const result = await classifyTeams(videoMetadata.video_id)  // POST /videos/{id}/classify-teams
setTeamClassifications(result.classifications)
setClassifySummary(result.summary)
```

### Backend

**File:** `routes/classification.py` → `classify_teams()` → `pipeline/team_classifier.py` → `classify_tracks()`

The classifier uses jersey colour. Ellistown wear a distinctive orange-yellow jersey (OpenCV HSV hue 14–28). Grass sits at hue 35–40 — a clean gap.

**Algorithm:**

1. For each track, select up to 30 evenly-spaced frames from across its full duration.
2. Build a `frame_idx → [(track_id, bbox), ...]` map, grouping all samples by frame. This ensures each frame is decoded exactly once regardless of how many tracks appear in it.
3. Single sequential forward pass through the video via `cv2.VideoCapture.set(CAP_PROP_POS_FRAMES, frame_idx)`:
   - For each sampled track in that frame, crop the top 50% of the bounding box (jersey region, avoids shorts/grass).
   - Convert crop to HSV; mask out low-saturation pixels (glare, shadows).
   - Count fraction of remaining pixels with H ∈ [14, 28] and S ≥ 100 (Ellistown yellow).
4. Aggregate: mean yellow fraction across all samples. If ≥ 15% → `"ellistown"`, else `"opposition"`. Confidence scales with how far above/below the threshold.

Output: `Dict[track_id, {team, confidence, mean_hsv}]` — stored to disk and in `store.team_classifications_cache`.

### Frontend display

**File:** `components/TeamClassificationPanel.tsx`

Shows jersey-colour HSV swatches (`hsvToCss()` converts OpenCV HSV to CSS), confidence bars, and a dropdown to override any classification. Override calls `PATCH /videos/{id}/classify-teams` immediately.

---

## Step 7 — Compute KPIs (Optional)

**File:** `components/ResultsViewer.tsx`, `components/KpiPanel.tsx`, `components/ClipSummaryCard.tsx`

The user clicks "Compute KPIs" (with an optional end-frame to trim the clip).

### Frontend

```typescript
const summary = await computeKpis(videoMetadata.video_id, clipEndFrame)
setKpiSummary(summary)
```

### Backend

**File:** `routes/kpi.py` → `compute_kpis()` → `pipeline/kpi.py` → `compute_clip_summary()`

All computation works on the in-memory `store.player_positions_cache[video_id]`. No video or frame decoding happens here — it's pure Python/NumPy on the position arrays.

#### Distance covered (per player)

```python
dx = np.diff(xs) / PX_PER_METRE    # PX_PER_METRE = 10 (px/m)
dy = np.diff(ys) / PX_PER_METRE
total_distance_m = np.sqrt(dx**2 + dy**2).sum()
```

Simple displacement sum between consecutive positions. Frame gaps (where interpolation spanned a detection gap) are handled correctly because we only sum spatial displacement, not speed × time.

#### Team spatial metrics (per frame)

For each frame in the clip, `compute_team_spatial()` computes per team:
- **Centroid:** mean x and mean y of all player positions in that frame (in meters).
- **Convex hull area (spread):** `scipy.spatial.ConvexHull(arr).volume` — in 2D, `volume` is actually the area. Zero if fewer than 3 players visible.
- **Centroid separation:** Euclidean distance between the two team centroids.

#### Zone balance (per frame)

The pitch is split into three equal thirds along the y-axis:
```
Defensive:  0 – 46.7m
Middle:    46.7 – 93.3m
Attacking: 93.3 – 140m
```
For each frame, count how many players from each team are in each zone.

#### Summary aggregation

Mean/min/max of centroid separation across all frames. Mean spread and mean centroid x/y per team.

### Frontend display

**File:** `utils/kpiUtils.ts` → `computeZoneAnalysis()`, `computeDepthSentence()`

`computeZoneAnalysis()` sums zone player counts across all frames and identifies the "detected zone" — whichever third had the most combined activity (used to highlight the relevant column in the zone balance table).

`computeDepthSentence()` computes a plain-English sentence describing how the relative team centroid depth changed from clip start to clip end (e.g. "Clip start: Opposition 8.3m goal-side · Clip end: Ellistown 2.1m goal-side").

**File:** `components/ClipSummaryCard.tsx` — plain-English 3-sentence summary of zone balance, team spread, and depth.

**File:** `components/KpiPanel.tsx` — full detailed table: centroid metrics, team spread, zone balance table, per-player distance covered chips.

---

## The Three Coordinate Systems

Understanding these is essential. Every bug that's ever been caused by wrong coordinates stems from mixing these up.

| Space | Description | x range | y range |
|---|---|---|---|
| **Image pixels** | Camera frame | 0..video_width (e.g. 0..1920) | 0..video_height (e.g. 0..1080) |
| **Pitch-canvas pixels** | Fixed output canvas (backend) | 0..850 | 0..1400 |
| **Pitch meters** | Physical pitch dimensions | 0..85 m | 0..140 m |
| **Display pixels** | Frontend canvas (40% scale) | 0..340 | 0..560 |

**Conversions:**
```
meters → canvas pixels:    multiply by 10  (PITCH_SCALE = 10 px/m)
canvas pixels → meters:    divide by 10
canvas pixels → display:   multiply by DISPLAY_SCALE (0.4)
image pixels → canvas:     apply homography H (the whole point of steps 4A–4B)
```

The homography H maps **image pixels** (camera space) → **pitch-canvas pixels**. It never touches meters or display pixels directly.

Annotations (`PitchPoint.x_img, y_img`) are stored in **image pixels** (original resolution). The pitch diagram in the annotator is drawn at display scale, but clicks are immediately up-scaled to original resolution before storing.

---

## The In-Memory Store and Disk Layout

**File:** `store.py`

```python
class VideoStore:
    videos:                     # UUID → {path, fps, num_frames, width, height, duration_seconds}
    detections_cache:           # UUID → List[Detection]
    v3_anchor_H_cache:          # UUID → {frame_idx: 3×3 ndarray}
    v3_per_frame_H_cache:       # UUID → {frame_idx: 3×3 ndarray}
    player_positions_cache:     # UUID → List[PlayerPitchPosition]
    team_classifications_cache: # UUID → {track_id: {team, confidence, mean_hsv}}
```

The store is a module-level singleton (`store = VideoStore()`). On server restart, only `store.videos` is restored from disk (via `restore_videos_from_disk()` in lifespan). All other caches start empty — each endpoint lazily loads from disk when needed:

```python
# Pattern used throughout routes/
detections = store.detections_cache.get(video_id) or load_detections(video_id)
```

The disk layout:
```
data/
  videos/
    {uuid}.mp4
    {uuid}_meta.json
  tracks/
    {uuid}.json                             ← List[Detection]
  annotations/
    {uuid}_annotations.json                 ← {frame_idx: {keypoints, lines}}
    {uuid}_v3_anchor_homographies.json      ← {str(frame_idx): [[3×3 list]]}
    {uuid}_v3_homographies.json             ← {str(frame_idx): [[3×3 list]]}
    {uuid}_team_classifications.json        ← {str(track_id): {team, confidence, mean_hsv}}
```

All disk I/O goes through `pipeline/persistence.py`. Never add `open()` calls in route files.

---

## Common Bugs and Where to Look

| Symptom | Likely cause | Where to look |
|---|---|---|
| Players appear at wrong pitch position | Wrong coordinate space | `map_players.py`: is `y_foot = det.y2` (bottom), not `det.y1` or `(y1+y2)/2`? |
| Systematic ~8px offset in annotations | Border vs outline on canvas | `AnchorFrameAnnotator.tsx`: canvas uses `outline:`, not `border:` |
| Anchor quality shows giant errors (>100px) | Hartley normalisation | `homography.py`: `_hartley_normalize` applied before building A matrix? |
| Trajectories jitter at every 25th frame | SG smoothing only on gaps | `trajectories.py`: smoothing must include original detected frames |
| No players visible in results | Tracking not run, or map_players not called | Check console log; run steps in order A→B→C→D |
| Team colours all wrong after classification | Yellow HSV threshold not matching jerseys | `team_classifier.py`: `YELLOW_HUE_MIN/MAX` tunable constants at top |
| Homography drifts badly in midfield | No line annotations in midfield | Add line annotations at 45m and halfway line for midfield anchor frames |
| Frontend canvas click coordinates off at zoom | `rect.width` calculation | `AnchorFrameAnnotator.tsx`: `x = (clientX - rect.left) * naturalWidth / rect.width` |
| Backend restart loses all positions | Not persisted | `store.player_positions_cache` is not saved to disk — re-run map+interpolate |
