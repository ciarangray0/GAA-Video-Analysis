# Detection Module

Covers `detect.py`, `video.py`, and `rendering.py`. Together these handle frame extraction, YOLO+BotSort tracking dispatch, and perspective warping.

---

## `video.py`

Simple OpenCV wrappers with no pipeline-specific logic.

### `get_video_metadata(video_path) → dict`
Opens the video with `cv2.VideoCapture`, reads `CAP_PROP_FPS`, `CAP_PROP_FRAME_COUNT`, `CAP_PROP_FRAME_WIDTH`, `CAP_PROP_FRAME_HEIGHT`, then closes immediately. Returns `{fps, num_frames, width, height, duration_seconds}`.

- If `fps` is 0 or negative (can happen with some encodings), defaults to 30.
- `duration_seconds` is rounded to 2 decimal places.
- `fps` is cast to `int` so it serialises cleanly to JSON.

### `extract_frame(video_path, frame_idx) → Optional[bytes]`
Seeks to `frame_idx` using `cap.set(CAP_PROP_POS_FRAMES, frame_idx)`, reads one frame, encodes it as JPEG at quality 85, and returns the bytes. Returns `None` if seek or read fails.

---

## `rendering.py`

### `warp_frame(img, H, out_w, out_h) → np.ndarray`
Single-line wrapper around `cv2.warpPerspective`. Takes a BGR frame and a 3×3 homography matrix, returns a new `out_w × out_h` BGR image that is the perspective-corrected bird's-eye view of the pitch.

This function is called by every warped-frame endpoint in `app.py`.

---

## `detect.py`

### `run_tracking(video_path) → List[Detection]`
Entry point for all tracking calls. Reads the `GPU_PROVIDER` environment variable:
- If `GPU_PROVIDER != "local"`: calls `_run_tracking_remote`.
- If `GPU_PROVIDER == "local"` (default): calls `_run_tracking_local` with a warning log.

### `_run_tracking_remote(video_path) → List[Detection]`
1. Calls `get_gpu_client()` from `gpu_inference/__init__.py` to get (or create) the singleton `GPUInferenceClient`.
2. Calls `client.track_video(video_path)`.

### `_run_tracking_local(video_path) → List[Detection]`
CPU fallback. Imports `ultralytics.YOLO` lazily (raises a helpful error if not installed). Calls `model.track(source, tracker="botsort.yaml", conf=0.35, device="cpu")`. Streams results frame by frame and converts each detection to a `Detection` object. The `class_name` is looked up from `model.names` dict.

**Note:** Local CPU inference is very slow on full-length videos. It is intended only for development/testing. The remote Modal GPU path is the production path.

---

## YOLO Model

The model (`v8s_960_v9.pt`) is a custom YOLOv8-small trained at 960px resolution on GAA footage. It outputs three classes:
- `"GAA-player-lablers"` — players
- `"Ball-labelers"` — the ball
- `"Refree-lablers"` — referees

Class names contain deliberate typos that match the training labels — these must not be corrected.

BotSort (included in Ultralytics) provides persistent track IDs across frames. The `tracker="botsort.yaml"` argument tells Ultralytics to use BotSort instead of the default ByteTrack.

---

## `GPU_PROVIDER` Environment Variable

| Value | Behaviour |
|-------|-----------|
| `"local"` (default) | Runs YOLO locally on CPU |
| `"modal"` | Sends video to Modal GPU service via HTTP |

When using Modal, `GPU_ENDPOINT_URL` must also be set to the deployed Modal endpoint URL (printed when running `modal deploy modal_yolo.py`).
