# Detection Module

This module handles three things: reading video files, running the YOLO+BotSort object tracker to find players, and converting a raw video frame into a bird's-eye pitch view. The code lives across three files: `video.py`, `rendering.py`, and `detect.py`.

---

## `video.py` — reading video files

This file contains simple utilities for opening a video and getting information out of it. There is no pipeline logic here — just wrappers around OpenCV's video-reading tools.

### `get_video_metadata(video_path) → dict`

**What it does:** Opens the video file, reads its properties, and returns them as a dictionary.

**Step by step:**

1. Open the file with `cv2.VideoCapture(video_path)`. This does not decode any frames — it just opens the container.
2. Read four properties from the video:
   - `CAP_PROP_FPS` — frames per second (e.g. 25.0 or 29.97)
   - `CAP_PROP_FRAME_COUNT` — total number of frames
   - `CAP_PROP_FRAME_WIDTH` and `CAP_PROP_FRAME_HEIGHT` — video resolution in pixels
3. Close the file immediately.
4. Return a dict with these values plus `duration_seconds` (rounded to 2 decimal places).

**Edge case:** Some video files report `fps=0` due to encoding quirks. If that happens, the code defaults to 30 fps rather than crashing or returning a nonsensical duration. The `fps` value is also cast to `int` (e.g. `29.97` becomes `29`) so it serialises cleanly to JSON.

---

### `extract_frame(video_path, frame_idx) → bytes or None`

**What it does:** Jumps to a specific frame in the video and returns it as a JPEG image.

**Step by step:**

1. Open the video with `cv2.VideoCapture`.
2. Jump to the requested frame using `cap.set(CAP_PROP_POS_FRAMES, frame_idx)`. Think of this like scrubbing to a specific timestamp — the video decoder jumps to the nearest keyframe and then decodes forward to the exact frame.
3. Call `cap.read()` to decode that one frame into a NumPy array of pixel values (in BGR format).
4. Encode it as a JPEG at quality 85 using `cv2.imencode`. Quality 85 is a good balance between file size and visual quality — quality 100 would be lossless but much larger.
5. Return the JPEG as a byte string. The API endpoint wraps this in an HTTP response so the browser can display it.

Returns `None` if the seek or read fails (e.g. the frame index is out of range).

---

## `rendering.py` — warping a frame

### `warp_frame(img, H, out_w, out_h) → image`

**What it does:** Takes a raw camera frame and a homography matrix, and produces a bird's-eye view of the pitch.

A **homography matrix** is a 3×3 grid of numbers that encodes the mathematical transformation between two planes — in this case, from the camera's perspective view to a flat top-down view of the pitch. The details of how that matrix is computed are in `homography.py`, but this function simply applies it.

`cv2.warpPerspective(img, H, (out_w, out_h))` does all the work. For every pixel in the output image, it works backwards through the transformation to find where that pixel came from in the original frame, and copies the colour. The output is `out_w × out_h` pixels — for this project, 850×1400, matching the pitch canvas dimensions.

This function is called by every warped-frame API endpoint.

---

## `detect.py` — running YOLO+BotSort

### `run_tracking(video_path) → List[Detection]`

**What it does:** Sends the video to a GPU server, runs YOLO+BotSort on every frame, and returns a flat list of `Detection` objects — one per bounding box per frame.

**Step by step:**

1. Call `get_gpu_client()` to get a `GPUInferenceClient` object. This client is a singleton — it is created once and reused. If the server connection is not yet established, this call creates it.
2. Call `client.track_video(video_path)`. This sends the video to the GPU server, which runs YOLO detection and BotSort tracking on every frame, and returns the results.

---

## The YOLO model

The model file is `v8s_960_v9.pt`. Breaking that name down:
- `v8s` — YOLOv8 "small" variant (fast, moderate accuracy)
- `960` — trained at 960px input resolution (higher resolution → better detection of small/distant players)
- `v9` — the ninth training run on GAA footage

It detects three object classes:

| Class string | What it is |
|--------------|------------|
| `"GAA-player-lablers"` | A player on the pitch |
| `"Ball-labelers"` | The ball |
| `"Refree-lablers"` | A referee |

**The typos are intentional.** These strings come directly from the YOLO model's internal class name list and must match exactly. The model was trained with labels that have these spellings (they were typos in the original training annotations). If you "correct" the spelling here, the string comparison will fail and all detections will be misclassified.

---

## What BotSort adds

YOLO alone detects objects frame by frame — each detection is independent. You get a list of bounding boxes per frame, but no way to know that box 3 in frame 50 is the same player as box 2 in frame 51.

**BotSort** is a tracker: it links detections across frames using motion prediction and appearance features, assigning each player a persistent `track_id` that stays the same for as long as the player is continuously visible. A `track_id` of `42`, for example, will refer to the same physical player across hundreds of frames.

BotSort is built into the Ultralytics library. Passing `tracker="botsort.yaml"` to the YOLO model switches it from the default ByteTrack algorithm to BotSort.

---

## GPU setup

Tracking is computationally expensive — running YOLO+BotSort on a 30-second video at 25fps means processing ~750 frames. This is done on a GPU server rather than the local machine.

The `GPU_PROVIDER` environment variable controls which server is used.

| Value | Behaviour |
|-------|-----------|
| `"modal"` | Sends the video to a Modal.com GPU service via HTTP |

When using Modal, you must also set `GPU_ENDPOINT_URL` to the URL of the deployed Modal endpoint. This URL is printed in the terminal when you run `modal deploy modal_yolo.py`.
