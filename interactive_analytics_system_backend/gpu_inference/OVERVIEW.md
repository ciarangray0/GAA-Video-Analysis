# GPU Inference Module

Covers `modal_yolo.py` (the Modal serverless GPU service) and `__init__.py` (the HTTP client that calls it from the backend).

---

## Architecture Overview

Running YOLO+BotSort inference on a full-length GAA video takes ~minutes on a GPU. The backend is hosted on Render (CPU-only). Rather than timeout or block, inference is offloaded to **Modal** — a serverless GPU platform.

```
Backend (Render, CPU)
  ↓  POST {video_base64}
Modal endpoint (GPU T4 container)
  ↓  returns JSON detections
Backend
  ↓  parses into List[Detection]
```

The video is base64-encoded and sent as a JSON body. This avoids multipart form data complexity. For a 500MB video the encoded payload is ~667MB — acceptable over a server-to-server connection but not from a browser.

---

## `modal_yolo.py` — The Modal Service

Deployed independently from the main FastAPI backend with `modal deploy modal_yolo.py`.

### Container Image: `yolo_image`
Built from `debian_slim(python_version="3.11")` with:
- System dependencies: libglib, libsm, libxext, ffmpeg, libgl (OpenCV requirements)
- Python packages: ultralytics, torch, torchvision, opencv-python-headless, numpy, lap (ByteTrack dependency), fastapi, pydantic

### Model Storage: `model_cache`
A `modal.Volume` named `"yolo-model-cache"` (created if missing). Mounted at `/model_cache`. The custom YOLO weights file `v8s_960_v9.pt` must be uploaded to this volume once:
```bash
modal volume put yolo-model-cache v8s_960_v9.pt /v8s_960_v9.pt
```

### `YOLOTracker` Class
Decorated with `@app.cls(gpu="T4", timeout=600, volumes={"/model_cache": model_cache}, scaledown_window=60)`.

- GPU: T4 (NVIDIA Tesla T4 — cheapest Modal GPU tier, ~16 GB VRAM, sufficient for YOLOv8-small at 960px).
- Timeout: 600 seconds (10 min). Long GAA halves can be 35 minutes of video at 25 fps ≈ 52,500 frames.
- `scaledown_window=60`: keep the container warm for 60 seconds after the last request. This avoids cold-start delay for back-to-back tracking jobs.

#### `load_model(self)` — `@modal.enter()`
Called once when the container starts. Loads the YOLO model from `/model_cache/v8s_960_v9.pt`. Raises `FileNotFoundError` with instructions if the model is not in the volume.

#### `_run_tracking(self, video_bytes) → List[Dict]`
Core inference method.

1. Writes `video_bytes` to a temp file (YOLO needs a file path, not bytes).
2. Calls `self.model.track(source=temp_path, tracker="botsort.yaml", imgsz=960, persist=True, conf=0.35, stream=True)`. `stream=True` reduces peak GPU memory by yielding results one frame at a time.
3. For each frame result, extracts `boxes.id`, `boxes.xyxy`, `boxes.conf`, `boxes.cls`.
4. Converts to plain dicts (JSON-serialisable).
5. Deletes the temp file in a `finally` block.
6. Returns a list of detection dicts.

#### `track_video_endpoint(self, request: TrackVideoRequest)` — `@modal.fastapi_endpoint(method="POST")`
The HTTP-facing endpoint. Decodes `request.video_base64`, calls `_run_tracking`, returns `{status, detections, count}` on success or `{status: "error", message, traceback}` on failure.

### `track_video_direct(video_bytes)` — `@app.function`
An alternative Modal function for direct SDK calls (not via HTTP). Not used in the current deployment — the HTTP endpoint is used instead.

---

## `__init__.py` — The HTTP Client

### `GPUProvider` enum
`MODAL`, `RUNPOD`, `LOCAL` — the provider backend supports. Only `MODAL` and `LOCAL` are currently wired up.

### `GPUInferenceClient`
An `httpx.Client`-based HTTP client with a 600-second timeout (matching the Modal container timeout).

#### `track_video(video_path) → List[Detection]`
Dispatches to `_track_modal` based on `self.provider`. Raises `ValueError` for `LOCAL` provider (callers should use `_run_tracking_local` in `detect.py` instead).

#### `_track_modal(video_path) → List[Detection]`
1. Reads the video file from disk.
2. Base64-encodes it: `base64.b64encode(video_bytes).decode("utf-8")`.
3. Logs the size being sent.
4. POSTs to `self.endpoint_url` with `{"video_base64": encoded}`.
5. Checks for `status == "error"` in the response and raises `RuntimeError` if found.
6. Converts each detection dict to a `Detection` object.

### `get_gpu_client() → GPUInferenceClient` (singleton)
Creates the singleton `GPUInferenceClient` on first call using environment variables:

| Variable | Required | Meaning |
|----------|----------|---------|
| `GPU_PROVIDER` | Yes | `"modal"` |
| `GPU_ENDPOINT_URL` | Yes | Full URL of the deployed Modal endpoint |
| `GPU_API_KEY` | No | Auth key (not used by Modal public endpoints) |

The endpoint URL is sanitised (whitespace/newlines stripped) since environment variables can pick up trailing newlines from `.env` files.
