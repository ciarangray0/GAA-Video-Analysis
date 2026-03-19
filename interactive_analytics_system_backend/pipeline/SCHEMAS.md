# Pipeline Schemas

All Pydantic models used for API request/response validation and for passing data between pipeline modules.

---

## Annotation Models (Request Inputs)

### `PitchPoint`
A single user-annotated correspondence between a point in the video frame and a named location on the pitch.

| Field | Type | Description |
|-------|------|-------------|
| `pitch_id` | `str` | Named vertex from `GAA_PITCH_VERTICES` (e.g. `"corner_tl"`) or the encoded format `line_<name>_x<X>_y<Y>` for points on pitch line segments |
| `x_img` | `float` | Horizontal pixel coordinate in the **original** video frame (0..naturalWidth) |
| `y_img` | `float` | Vertical pixel coordinate in the **original** video frame (0..naturalHeight) |

Coordinates are stored in original-image space, not in the scaled canvas used for display. This is important: the annotator canvas may be scaled down for display, but the click formula maps clicks back to natural pixel coords before creating a `PitchPoint`.

---

### `LineAnnotation`
A user-annotated line: the user clicks two points that both lie on a known pitch line. The system samples N points along the segment and adds them as 1D DLT constraints.

| Field | Type | Description |
|-------|------|-------------|
| `line_id` | `str` | Key from `GAA_PITCH_LINES` (horizontal) or `GAA_PITCH_SIDELINES` (vertical), e.g. `"45m_top"`, `"left_sideline"` |
| `u1`, `v1` | `float` | First endpoint in image pixels |
| `u2`, `v2` | `float` | Second endpoint in image pixels |

The constraint provided is: every point on this line segment shares the **same Y-value** (horizontal lines) or **same X-value** (vertical lines) on the pitch canvas. This is a 1D constraint, not the full 2D constraint from a keypoint.

---

### `AnchorFrameAnnotation`
Complete annotation for one anchor frame — sent as the request body to `POST /homographies/v3`.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `frame_idx` | `int` | — | 0-based frame index |
| `points` | `List[PitchPoint]` | — | Keypoint correspondences (need ≥ 4) |
| `lines` | `List[LineAnnotation]` | `[]` | Optional line constraints; empty list is fine |

The default `lines=[]` means old saved annotations (without line data) still deserialise correctly.

---

## Detection Model

### `Detection`
One YOLO+BotSort detection for a single bounding box in a single frame.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `frame_idx` | `int` | — | Frame number (0-based) |
| `track_id` | `int` | — | BotSort persistent track ID |
| `x1`, `y1`, `x2`, `y2` | `float` | — | Bounding box (image pixels, xyxy format) |
| `confidence` | `float` | — | YOLO detection confidence 0–1 |
| `class_name` | `str` | `CLASS_PLAYER` | Detection class string |

The `class_name` default (`CLASS_PLAYER = "GAA-player-lablers"`) exists so that JSON files saved before class names were added still deserialise correctly as players.

---

## Class Name Constants

```python
CLASS_PLAYER  = "GAA-player-lablers"   # default player class from the YOLO model
CLASS_BALL    = "Ball-labelers"
CLASS_REFEREE = "Refree-lablers"        # note: deliberate typo matches the model's label
```

These strings come directly from the YOLO model's `names` dict and must match exactly. They are used in `filter_detections_for_mapping` to separate players, the ball, and referees.

---

## Player Position Model

### `PlayerPitchPosition`
One player's mapped position on the pitch canvas for a single frame.

| Field | Type | Description |
|-------|------|-------------|
| `frame_idx` | `int` | Frame index |
| `track_id` | `int` | BotSort track ID |
| `x_pitch` | `float` | Horizontal position in **pitch-canvas pixels** (0..850) |
| `y_pitch` | `float` | Vertical position in **pitch-canvas pixels** (0..1400) |
| `source` | `str` | `"homography"` (anchor frame), `"homography_interp"` (propagated frame), or `"interpolated"` (filled by trajectory interpolation) |

The `source` field lets the frontend distinguish high-confidence anchor-frame positions from propagated ones, and from fully synthetic interpolated positions.

---

## Response Models

### `VideoCreateResponse`
Returned by `POST /videos`.

| Field | Description |
|-------|-------------|
| `video_id` | UUID string for all subsequent API calls |
| `fps` | Frames per second (int) |
| `num_frames` | Total frame count |
| `width`, `height` | Video dimensions in pixels |
| `duration_seconds` | Video length |

### `TrackResponse`
Returned by `POST /videos/{id}/track`.

| Field | Description |
|-------|-------------|
| `frames_processed` | Index of the last frame that had ≥1 detection + 1 |
| `tracks` | Number of unique BotSort track IDs found |

### `HomographyResponse`
Defines `frames: List[int]` — the anchor frame indices for which homographies were computed. Not returned directly by v3 (which returns a richer dict), but serves as a type reference.

### `InterpolationResponse`
Returned by `POST /videos/{id}/interpolate`.

| Field | Description |
|-------|-------------|
| `frames_generated` | Number of position objects with `source="interpolated"` that were created |
| `method` | Always `"linear"` for the current implementation |

---

## `PitchAnnotation` (legacy)
`frame_idx: int` + `points: List[PitchPoint]` — defined but no longer used by any active endpoint. Kept in the file as a type reference.
