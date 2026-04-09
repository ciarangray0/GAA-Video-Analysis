# Pipeline Schemas (`schemas.py`)

This file defines all the **data shapes** used by the API and the pipeline. Think of each schema as a contract: "when this data comes in, it must look exactly like this; when this data goes out, it will look exactly like this."

The schemas are written using **Pydantic**, a Python library that automatically validates data and raises a clear error if something is the wrong type or is missing. For example, if the API receives a JSON body for a homography request but `frame_idx` is a string instead of an integer, Pydantic catches that before any pipeline code runs.

---

## Annotation models (inputs from the frontend)

These models represent data that the user creates by clicking on the annotator in the frontend.

### `PitchPoint`

A single correspondence point: "this pixel in the video frame corresponds to this named location on the pitch."

| Field | Type | What it stores |
|-------|------|----------------|
| `pitch_id` | `str` | A named location like `"corner_tl"` (top-left corner) from the list of known pitch vertices |
| `x_img` | `float` | Horizontal pixel position in the **original video frame** |
| `y_img` | `float` | Vertical pixel position in the **original video frame** |

**Why original image pixels?** The annotator canvas is displayed at a scaled-down size on screen (e.g. the video is 1920px wide but the canvas is 1000px wide). If the coordinates were stored in canvas pixels and the canvas size ever changed, all old annotations would be wrong. Storing in original image pixels means the coordinates remain valid regardless of how the UI is displayed.

---

### `LineAnnotation`

A line constraint: the user clicks two points that both lie on the same known pitch line (like the 45-metre line). This gives the homography solver extra information — it does not need to know the exact position along the line, only which line the points are on.

| Field | Type | What it stores |
|-------|------|----------------|
| `line_id` | `str` | The name of the pitch line, e.g. `"45m_top"` or `"left_sideline"` |
| `u1`, `v1` | `float` | First clicked point (image pixels) |
| `u2`, `v2` | `float` | Second clicked point (image pixels) |

The pipeline samples several evenly-spaced points along this segment and adds a 1D constraint for each: every point on a horizontal line shares the same Y coordinate on the pitch canvas. This is a weaker constraint than a keypoint (which fixes both X and Y) but is much easier for the user to annotate accurately.

---

### `AnchorFrameAnnotation`

The complete annotation for one anchor frame. This is the request body sent to `POST /homographies/v3`.

| Field | Type | Default | What it stores |
|-------|------|---------|----------------|
| `frame_idx` | `int` | — | Which frame was annotated (0-based index into the video) |
| `points` | `List[PitchPoint]` | — | The keypoint correspondences (minimum 4 needed to solve a homography) |
| `lines` | `List[LineAnnotation]` | `[]` | Optional line constraints |

The default value of `[]` for `lines` is important: it means annotation files that were saved before line constraints were added to the system can still be loaded and used without error — they just have no line data.

---

## Detection model

### `Detection`

One detection from YOLO+BotSort: a single bounding box around a player, ball, or referee in a single frame.

| Field | Type | Default | What it stores |
|-------|------|---------|----------------|
| `frame_idx` | `int` | — | Which frame this detection appears in |
| `track_id` | `int` | — | BotSort's persistent ID for this player across frames |
| `x1`, `y1`, `x2`, `y2` | `float` | — | Bounding box corners in image pixels (top-left to bottom-right) |
| `confidence` | `float` | — | YOLO's confidence score, 0.0 to 1.0 |
| `class_name` | `str` | `CLASS_PLAYER` | What object was detected |

**Why does `class_name` have a default?** Detection data is saved to disk as JSON after tracking. Before class names were added to the schema, those JSON files have no `class_name` field. The default (`CLASS_PLAYER`) means old files load correctly — missing class names are assumed to be player detections.

---

## Class name constants

```python
CLASS_PLAYER  = "GAA-player-lablers"
CLASS_BALL    = "Ball-labelers"
CLASS_REFEREE = "Refree-lablers"
```

These strings come directly from the YOLO model's internal class names and must match exactly — including the typos. The model was trained with labels that have these spellings, so if you "fix" the typo here, the matching will break and nothing will be classified correctly.

These constants are used throughout the pipeline to filter detections — for example, `filter_detections_for_mapping` in `map_players.py` uses them to separate players from the ball and referees before running the homography mapping.

---

## Player position model

### `PlayerPitchPosition`

One player's mapped position on the pitch canvas for a single frame. This is the core output of the player-mapping pipeline.

| Field | Type | What it stores |
|-------|------|----------------|
| `frame_idx` | `int` | Which frame this position is for |
| `track_id` | `int` | Which player (BotSort track ID) |
| `x_pitch` | `float` | Horizontal position in **pitch-canvas pixels**, 0–850 |
| `y_pitch` | `float` | Vertical position in **pitch-canvas pixels**, 0–1400 |
| `source` | `str` | How this position was produced |

**The `source` field** records the confidence level of the position:
- `"homography"` — computed directly from an anchor frame that the user annotated. Most reliable.
- `"homography_interp"` — computed via the per-frame propagated homography (optical flow was used to extend an anchor frame's homography to a nearby frame). Slightly less reliable.
- `"interpolated"` — the player was not detected in this frame and the position was filled in synthetically by linear interpolation between known positions. Least reliable — treat as an estimate.

The frontend can use `source` to shade player dots differently, or to filter out interpolated positions from KPI computations.

---

## Response models

These models define what the API sends back after a successful request.

### `VideoCreateResponse`

Returned by `POST /videos` after a video is uploaded.

| Field | What it contains |
|-------|-----------------|
| `video_id` | A UUID string — used in every subsequent API call (e.g. `/videos/{video_id}/track`) |
| `fps` | Frames per second, as an integer |
| `num_frames` | Total frame count |
| `width`, `height` | Video dimensions in pixels |
| `duration_seconds` | How long the video is |

---

### `TrackResponse`

Returned by `POST /videos/{id}/track` after YOLO+BotSort has processed the video.

| Field | What it contains |
|-------|-----------------|
| `frames_processed` | The index of the last frame that had at least one detection, plus 1 |
| `tracks` | How many unique BotSort track IDs were found |

---

### `HomographyResponse`

Defines `frames: List[int]` — the anchor frame indices that have computed homographies. Not returned directly by the v3 endpoint (which returns a richer response), but used as a type reference elsewhere in the codebase.

---

### `InterpolationResponse`

Returned by `POST /videos/{id}/interpolate` after trajectory interpolation has run.

| Field | What it contains |
|-------|-----------------|
| `frames_generated` | How many new position entries with `source="interpolated"` were created |
| `method` | Always `"linear"` — the current implementation uses linear interpolation |

---

## `PitchAnnotation` (legacy)

A simpler annotation model: `frame_idx` + `points: List[PitchPoint]`, with no line data. Defined in the file but not used by any active endpoint. Kept as a reference — removing it would not affect any running code.

---

## Team classification constants

### `VALID_TEAMS`

```python
VALID_TEAMS = {"ellistown", "opposition", "referee", "ignore"}
```

This is a Python **set** (a collection of unique values). When the PATCH endpoint receives a team override request, it checks `new_team in VALID_TEAMS` before doing anything. If the value is not in this set — for example if someone sends `"home_team"` — the API returns HTTP 400 (Bad Request) immediately.

---

### `TeamOverrideRequest`

Request body for `PATCH /videos/{id}/classify-teams`. Used when the automatic jersey-colour classifier gets a track wrong and the user wants to correct it.

| Field | Type | What it stores |
|-------|------|----------------|
| `track_id` | `int` | The BotSort track ID to reassign |
| `team` | `str` | The new team string — must be one of `VALID_TEAMS` |
