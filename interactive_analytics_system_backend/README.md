# GAA Video Analysis — Backend

FastAPI backend for the GAA video analysis pipeline. Accepts an uploaded MP4, runs
YOLO+BotSort player tracking via a Modal serverless GPU, computes per-frame
perspective-correcting homographies from user-supplied pitch annotations, maps every
player detection to a fixed 2-D pitch canvas, and computes spatial KPIs.

---

## Quick Start

### 1. Deploy the Modal GPU service (required for tracking)

```bash
pip install modal
modal token new
modal volume put yolo-model-cache v8s_960_v9.pt /v8s_960_v9.pt
modal deploy gpu_inference/modal_yolo.py
# Copy the printed endpoint URL → GPU_ENDPOINT_URL env var
```

### 2. Run locally

```bash
cd interactive_analytics_system_backend
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

export GPU_PROVIDER=modal
export GPU_ENDPOINT_URL=https://your-modal-endpoint.modal.run

uvicorn main:app --reload --port 8000
```

Interactive docs: http://localhost:8000/docs

### 3. Deploy to Render

Connect your GitHub repo to Render; the `render.yaml` in this directory is
pre-configured. Set `ALLOWED_ORIGINS`, `GPU_PROVIDER`, and `GPU_ENDPOINT_URL` in
the Render dashboard.

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `GPU_PROVIDER` | `modal` | GPU inference provider (`modal`) |
| `GPU_ENDPOINT_URL` | — | URL of the deployed Modal endpoint |
| `ALLOWED_ORIGINS` | `*` | CORS allow-list (comma-separated) |
| `MAX_VIDEO_SIZE_MB` | `500` | Maximum upload size |
| `DATA_DIR` | `data` | Root directory for all persisted files |

---

## Project Structure

```
interactive_analytics_system_backend/
├── app.py                   # Creates FastAPI app, registers all routers
├── main.py                  # Uvicorn entry point
├── store.py                 # VideoStore singleton — in-memory state
├── routes/                  # HTTP endpoint handlers (one file per domain)
│   ├── deps.py              # Shared dependency: get_video_or_404
│   ├── videos.py            # Upload, frame, warped-frame endpoints
│   ├── detection.py         # POST /track, GET /detections endpoints
│   ├── homography.py        # POST /homographies/v3, anchor-quality endpoints
│   ├── mapping.py           # map_players, interpolate, players endpoints
│   ├── classification.py    # classify-teams + PATCH override endpoints
│   └── kpi.py               # compute-kpis endpoint
├── pipeline/                # Pure processing logic — no HTTP concerns
│   ├── config.py            # OUT_W, OUT_H, PITCH_SCALE constants
│   ├── gaa_pitch_config.py  # GAA pitch geometry: vertices, lines, sidelines
│   ├── schemas.py           # Pydantic models for all data types
│   ├── persistence.py       # All disk I/O (save/load JSON, homographies, etc.)
│   ├── video.py             # OpenCV: metadata extraction, frame extraction
│   ├── rendering.py         # warp_frame (cv2.warpPerspective wrapper)
│   ├── homography.py        # v3 DLT anchor H computation
│   ├── line_constraints.py  # DLT line constraint helpers
│   ├── constrained_homography.py  # LK optical flow per-frame propagation
│   ├── map_players.py       # Filter detections, map bbox → pitch coords
│   ├── trajectories.py      # Interpolation + SG smoothing + velocity clamp
│   ├── team_classifier.py   # Jersey-colour HSV team classification
│   └── kpi.py               # Spatial KPI computation
├── gpu_inference/           # Remote GPU client
│   ├── __init__.py          # GPUInferenceClient (httpx wrapper for Modal)
│   ├── modal_yolo.py        # Modal serverless GPU service (deploy this)
│   └── README.md            # GPU setup instructions
├── tests/                   # pytest suite
├── models/                  # YOLO model weights (v8s_960_v9.pt)
├── requirements.txt
└── render.yaml              # Render deployment config
```

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Liveness check |
| POST | `/videos` | Upload MP4, extract metadata |
| GET | `/videos/{id}/frame/{idx}` | Raw video frame as JPEG |
| GET | `/videos/{id}/frames/{idx}/warped` | Warped pitch view (bird's-eye) |
| GET | `/videos/{id}/frames/{idx}/detections_overlay` | Raw frame + BotSort boxes |
| GET | `/videos/{id}/detections` | All raw YOLO+BotSort detections |
| POST | `/videos/{id}/track` | Run YOLO+BotSort (dispatches to Modal) |
| POST | `/videos/{id}/homographies/v3` | Compute anchor Hs + propagate per-frame |
| GET | `/line-constraints/available-lines` | Line IDs usable for annotations |
| GET | `/videos/{id}/homographies/anchor-quality` | Per-keypoint reprojection quality |
| POST | `/videos/{id}/map_players` | Map detections → pitch canvas coords |
| POST | `/videos/{id}/interpolate` | Interpolate + smooth trajectories |
| GET | `/videos/{id}/players` | All player positions (sparse + interpolated) |
| POST | `/videos/{id}/classify-teams` | Classify tracks by jersey colour |
| GET | `/videos/{id}/classify-teams` | Return stored team classifications |
| PATCH | `/videos/{id}/classify-teams` | Override a single track's team |
| POST | `/videos/{id}/compute-kpis` | Compute spatial KPIs; `?end_frame=N` trims clip |

---

## Running Tests

```bash
cd interactive_analytics_system_backend
pytest -v
```

70 tests pass. Known pre-existing failures (unrelated to current code):
- `test_homography.py` — references a deleted function
- `test_line_constraints.py` — `test_validate_tilted_line_fails` (logic under review)
- `test_trajectories.py` — one edge case
- `test_gpu_inference.py` — requires Modal credentials
