# `lib/` Overview

Three TypeScript modules providing API calls, canvas drawing, and shared constants.

---

## `api.ts`

All backend API calls. Every function throws a typed `Error` on HTTP failure (parses `err.detail` from FastAPI's error response format).

| Function | Endpoint | Description |
|----------|----------|-------------|
| `uploadVideo(file)` | `POST /videos` | Uploads the MP4 via `FormData`, returns `VideoMetadata` |
| `trackVideo(videoId)` | `POST /videos/{id}/track` | Triggers YOLO+BotSort, returns `{frames_processed, tracks}` |
| `getDetections(videoId)` | `GET /videos/{id}/detections` | Returns all raw detection objects; returns `[]` on error (non-throwing) |
| `computeHomographiesV2(videoId, annotations)` | `POST /videos/{id}/homographies/v2` | Legacy v2 endpoint — still in `api.ts` but PipelineSteps calls the v3 endpoint directly via `apiFetch`. |
| `mapPlayers(videoId)` | `POST /videos/{id}/map_players` | Returns `PlayerPosition[]` |
| `interpolateTrajectories(videoId, start, end, params)` | `POST /videos/{id}/interpolate?...` | Builds query string from `InterpolationParams`, returns `{frames_generated, method}` |
| `getPlayerPositions(videoId)` | `GET /videos/{id}/players` | Returns all player positions (sparse + interpolated) |
| `classifyTeams(videoId)` | `POST /videos/{id}/classify-teams` | Runs jersey-colour classification; returns `ClassifyTeamsResponse` (`{classifications, summary}`) |
| `getTeamClassifications(videoId)` | `GET /videos/{id}/classify-teams` | Returns stored `TeamClassifications` dict; returns `{}` on 404 (non-throwing) |
| `overrideTeamClassification(videoId, trackId, team)` | `PATCH /videos/{id}/classify-teams` | Overrides one track's team assignment; returns updated `TeamClassifications` |

**`API_URL`** is read from `process.env.NEXT_PUBLIC_API_URL` at build time, defaulting to `http://localhost:8000`.

**`InterpolationParams` interface:**
```typescript
{ sgLongWindow?: number; sgMidWindow?: number; maxVelPx?: number }
```
Undefined params are omitted from the query string.

---

## `pitch.ts`

Canvas drawing helpers for the pitch diagram (annotation UI) and the results pitch view.

### `pitchToCanvas(pitchX, pitchY) → {x, y}`
Converts pitch coordinates in meters to display canvas pixels:
```typescript
x = (pitchX / GAA_PITCH_WIDTH)  * PITCH_DISPLAY_WIDTH    // = pitchX * (340/85) ≈ pitchX * 4
y = (pitchY / GAA_PITCH_LENGTH) * PITCH_DISPLAY_HEIGHT   // = pitchY * (560/140) = pitchY * 4
```

### `drawHorizontalLinePair(ctx, y1, y2, width)` (internal)
Draws two horizontal lines in a single `beginPath`/`stroke` for efficiency. Used by both pitch drawing functions to draw symmetric yard line pairs.

### `drawPitchDiagram(canvas, anchorFrames, currentAnchorIdx, pendingFrameClick, pendingLinePoint1)`
Draws the pitch diagram for the annotation UI.

1. Sets canvas to `PITCH_DISPLAY_WIDTH × PITCH_DISPLAY_HEIGHT`, fills green background, draws white border.
2. Draws the halfway line, 4 symmetric horizontal line pairs (13m/127m, 20m/120m, 45m/95m, 65m/75m) using `drawHorizontalLinePair`.
3. Draws 13m box vertical lines (x=33m, x=52m from each endline to the 13m line).
4. Draws goalie box (x=35.5m–49.5m, depth 4.5m from each endline).
5. If `pendingFrameClick`: highlights all `PITCH_LINE_SEGMENTS` in semi-transparent yellow (showing the user what can be clicked), draws all vertices as yellow circles.
6. Otherwise: draws all vertices as orange circles (available).
7. Annotated vertices (in `currentAnchor.points`) are drawn green.
8. If `pendingFrameClick`: draws an overlay text box at the bottom showing the pending image coordinates and instructions.

### `drawPitch(canvas, positions, frame, teamClassifications?)`
Draws the results pitch view. The optional `teamClassifications` argument is a `TeamClassifications` dict (keyed by `track_id.toString()`).

1. Resets canvas to `PITCH_DISPLAY_WIDTH × PITCH_DISPLAY_HEIGHT`, fills green background, draws white border inset 2px.
2. Draws all pitch markings at `rgba(255,255,255,0.55)` opacity.
3. Draws 20m semicircles (radius = 13/140 × HEIGHT, curving into pitch from the 20m lines).
4. **Ghost dots:** for each track that appeared in some past frame but is absent from the current frame, finds the most recent position and draws a grey semi-transparent dot. Ghost dots are drawn first (underneath active dots).
5. **Active player dots:** for each position at `frame`, draws a coloured filled circle (radius 8px) with a white stroke (red stroke if out-of-bounds). Track ID printed inside.

**Colour logic (`getPlayerColor`):**
- If `teamClassifications` is provided and the track has `team === 'referee'` or `team === 'ignore'`: returns `null` — the track is hidden entirely.
- If `team === 'ellistown'`: returns `'#FFD700'` (gold).
- If `team === 'opposition'`: returns `'#4488FF'` (blue).
- Otherwise (no classification or unrecognised team): falls back to the golden-angle HSL scheme:
  ```typescript
  `hsl(${(trackId * 137.508) % 360}, 70%, 50%)`
  ```
  Multiplying by 137.508° (the golden angle) ensures adjacent track IDs get maximally different hues.

6. Frame/player count overlay (top-left corner).

### `hsvToCss(h_cv, s_cv, v_cv) → string`
Converts OpenCV HSV values (H 0–179, S 0–255, V 0–255) to a CSS `rgb()` string. Used by `ResultsViewer` to render jersey-colour swatches in the team classification panel. Exported alongside `drawPitch`.

---

## `constants.ts`

All pitch geometry constants used in the frontend.

| Export | Value | Description |
|--------|-------|-------------|
| `PITCH_CANVAS_W` | `850` | Backend pitch canvas width (px) |
| `PITCH_CANVAS_H` | `1400` | Backend pitch canvas height (px) |
| `DISPLAY_SCALE` | `0.4` | Scale factor for frontend pitch display |
| `PITCH_DISPLAY_WIDTH` | `340` | Frontend display width (= 850 × 0.4) |
| `PITCH_DISPLAY_HEIGHT` | `560` | Frontend display height (= 1400 × 0.4) |
| `GAA_PITCH_WIDTH` | `85.0` | Pitch width in meters |
| `GAA_PITCH_LENGTH` | `140.0` | Pitch length in meters |
| `AVAILABLE_LINES` | Record | Line IDs with labels, y_meters (horizontal) or x_meters + orientation (vertical) |
| `GAA_PITCH_VERTICES` | Record | Named vertex positions `[x_m, y_m]` |
| `PITCH_LINE_SEGMENTS` | Array | Line segments `{name, x1, y1, x2, y2}` used for point-on-line annotations |

### `AVAILABLE_LINES`
Extends `GAA_PITCH_LINES` with display labels and orientation information:
```typescript
'45m_top': { label: '45m Line (Top)', y_meters: 45.0 }
'left_sideline': { label: 'Left Sideline', x_meters: 0.0, orientation: 'vertical' }
```
The `orientation` field is used in `AnchorFrameAnnotator` to colour line annotations (cyan for horizontal, orange for vertical).

### `PITCH_LINE_SEGMENTS`
17 named line segments covering all pitch markings. Used by `AnchorFrameAnnotator` to support clicking *anywhere on a line* (not just at a named vertex) to create a line-segment keypoint with an encoded `pitch_id` like `"line_45m_top_x42.5_y45.0"`.

### `GAA_PITCH_VERTICES`
Mirror of the backend `GAA_PITCH_VERTICES` dict, typed as `Record<string, [number, number]>`. Used by `AnchorFrameAnnotator` for pitch diagram vertex snapping and by `drawPitchDiagram` for drawing selectable dots.
