# `ResultsViewer` Component

Displays the processed results: a side-by-side view of the original video and a 2D pitch canvas with player positions. Supports playback, frame stepping, speed control, BotSort overlay toggle, team classification, KPI computation, and an analysis trim slider to exclude trailing frames from pitch display and KPI computation.

---

## Props

| Prop | Description |
|------|-------------|
| `videoMetadata` | fps, num_frames — used for frame/time conversion |
| `videoFile` | The raw `File` object — used to create a blob URL for the video element |
| `playerPositions` | All `PlayerPosition` objects (sparse + interpolated) |
| `currentFrame` | Currently displayed frame (controlled by parent) |
| `onFrameChange` | Called when the frame changes (slider, playback, step buttons) |
| `processedStartFrame, processedEndFrame` | The interpolated range — playback stops at `endFrame` |
| `homographyFrameIndices` | Anchor frame indices — shown in the mapping debug panel |
| `processedFps` | FPS for frame↔time conversion during playback |

---

## Local State

| State | Description |
|-------|-------------|
| `isPlaying` | Whether the RAF playback loop is active |
| `playbackSpeed` | Current speed multiplier (0.25, 0.5, 1, 2, 4×) |
| `showBotSortOverlay` | Toggle for the BotSort bounding-box overlay image |
| `videoObjectUrl` | Blob URL created from `videoFile` |
| `showMappingView` | Whether the warped-frame debug panel is open |
| `teamClassifications` | `TeamClassifications` dict (track_id → `{team, confidence, mean_hsv}`) — empty until "Classify Teams" is run |
| `classifySummary` | `ClassifyTeamsSummary` from the most recent `POST /classify-teams` response, or `null` |
| `isClassifying` | True while the classify-teams API call is in flight |
| `classifyError` | Error string if the last classification attempt failed, otherwise `null` |
| `kpiSummary` | `KpiSummary` returned by `POST /compute-kpis`, or `null` until computed |
| `isComputingKpis` | True while the compute-kpis API call is in flight |
| `kpiError` | Error string if the last KPI computation failed, otherwise `null` |
| `trimEndFrame` | **Committed** analysis trim end frame. All pitch display, player badges, playback, and KPI computation use positions ≤ this value. Synced to `processedEndFrame` when the prop changes. |
| `trimDragFrame` | **Live** slider position while dragging the trim slider. Updates on every `onChange` event for smooth display. Does not trigger canvas redraws or KPI changes — only the label updates. Committed to `trimEndFrame` when the user clicks "Apply trim". |

---

## Refs

| Ref | Description |
|-----|-------------|
| `canvasRef` | The pitch canvas element |
| `videoPlayerRef` | The HTML `<video>` element |
| `animFrameRef` | `requestAnimationFrame` ID — cancelled on unmount |

---

## Video Blob URL

```typescript
useEffect(() => {
  const url = URL.createObjectURL(videoFile)
  setVideoObjectUrl(url)
  return () => URL.revokeObjectURL(url)
}, [videoFile])
```

A blob URL is created from the `File` object rather than uploading the video again. This avoids a second HTTP request and works offline after upload. The URL is revoked on unmount to free memory.

---

## `analysisPositions` (useMemo)

```typescript
const analysisPositions = useMemo(
  () => playerPositions.filter(p => p.frame_idx <= trimEndFrame),
  [playerPositions, trimEndFrame]
)
```

A filtered view of `playerPositions` that respects the trim end frame. All pitch rendering, player badges, `getFramesWithPositions`, and playback stop-check use `analysisPositions` rather than the raw `playerPositions` prop. KPI computation passes `trimEndFrame` to the backend as `?end_frame=N` so the server-side filtering matches. The raw `playerPositions` prop (from the parent) is never mutated — trimming is purely a read-time filter.

---

## `getFramesWithPositions() → number[]`

Returns a sorted array of all frame indices in `analysisPositions` that have at least one player. Used by `goToFrame` and `skipFrames` to navigate only to frames with data.

---

## `goToFrame(frameIdx)`

Snaps to the nearest frame that has player position data:
```typescript
let nearest = frames[0]
let minDist = Math.abs(frameIdx - nearest)
for (const f of frames) {
  const dist = Math.abs(frameIdx - f)
  if (dist < minDist) { minDist = dist; nearest = f }
}
onFrameChange(nearest)
if (!isPlaying && videoPlayerRef.current) {
  videoPlayerRef.current.currentTime = nearest / videoMetadata.fps
}
```

This handles the slider: the user may drag to a frame that has no positions (e.g. a gap in tracking), and `goToFrame` snaps to the nearest valid frame.

---

## `startPlayback()`

```typescript
video.playbackRate = playbackSpeed
// Reset to start if ended or past the trim end
if (video.ended || video.currentTime >= trimEndFrame / fps) {
  video.currentTime = startTime
}
setIsPlaying(true)
video.play()
  .then(() => {
    animFrameRef.current = requestAnimationFrame(onPlaybackFrame)
  })
  .catch(err => {
    console.warn('Playback blocked:', err)
    setIsPlaying(false)
  })
```

`video.play()` returns a Promise that rejects if the browser blocks autoplay. The `.then()` call ensures the RAF loop only starts once the video is actually playing — starting RAF before the video plays would result in the pitch canvas updating but the video frame not moving.

The `video.ended` check handles the case where the user clicks play after the video has played to the end — without this, `play()` would resume from the end and immediately trigger the ended state again.

---

## `onPlaybackFrame()`

The RAF callback. Called ~60 times per second while playing.

```typescript
const fps = processedFps || videoMetadata.fps || 25
const frameIdx = Math.round(video.currentTime * fps)
if (frameIdx > trimEndFrame) {
  video.pause(); setIsPlaying(false); return
}
onFrameChange(frameIdx)
animFrameRef.current = requestAnimationFrame(onPlaybackFrame)
```

Converts the video's `currentTime` to a frame index using `processedFps`. Stops playback at `trimEndFrame` (the committed trim end, defaulting to `processedEndFrame`) even if the video continues beyond. Calls `onFrameChange` to update `currentFrame` in parent, which triggers the pitch canvas redraw.

---

## `stopPlayback()`

Cancels the RAF loop and pauses the video.

---

## Video–Pitch Sync Effect

```typescript
useEffect(() => {
  if (!isPlaying && video.readyState >= 2) {
    const timeInSeconds = currentFrame / videoMetadata.fps
    if (Math.abs(video.currentTime - timeInSeconds) > 0.1) {
      video.currentTime = timeInSeconds
    }
  }
}, [currentFrame, isPlaying, ...])
```

When not playing, the video is always kept in sync with `currentFrame`. The 0.1-second threshold prevents unnecessary seeks when the video is already close (small floating-point drift from the playback loop). Only runs when `readyState >= 2` (video has enough data to seek). There is no user-togglable sync mode — sync is always active.

---

## `drawPitch` Effect

```typescript
useEffect(() => {
  if (canvasRef.current && analysisPositions.length > 0) {
    drawPitch(canvasRef.current, analysisPositions, currentFrame, teamClassifications, showTrails)
  }
}, [currentFrame, analysisPositions, teamClassifications, showTrails])
```

Redraws the entire pitch canvas whenever `currentFrame`, `analysisPositions`, or `teamClassifications` changes. Uses `analysisPositions` (not the raw `playerPositions` prop) so the pitch canvas automatically respects the trim end frame — positions beyond the trim are never drawn. `drawPitch` accepts an optional `teamClassifications` argument — when provided, dots are coloured by team (yellow for Ellistown, blue for opposition) rather than the default golden-angle HSL scheme. Tracks classified as `'referee'` or `'ignore'` are hidden. See `lib/OVERVIEW.md` for the full `drawPitch` implementation.

---

## BotSort Overlay

When `showBotSortOverlay` is true, renders an `<img>` fetching `GET /videos/{id}/frames/{frame}/detections_overlay`. The `key={currentFrame}` prop forces the image to reload when the frame changes, since the `src` URL changes too. An `onError` handler hides the image and shows a fallback message if no detections are available.

---

## Debug Coordinate Table

Always visible below the main view. For the current frame, lists every player's `x_pitch`, `y_pitch`, computed display coordinates, source, and an OK/OUT status.

```typescript
const xDisplay = (pos.x_pitch / PITCH_CANVAS_W) * PITCH_DISPLAY_WIDTH
const yDisplay = (pos.y_pitch / PITCH_CANVAS_H) * PITCH_DISPLAY_HEIGHT
const isOutOfBounds = pos.x_pitch < 0 || pos.x_pitch > PITCH_CANVAS_W || ...
```

Out-of-bounds rows are highlighted in red. This was essential during development to diagnose homography and coordinate system bugs.

---

## Team Classification

A "Classify Teams" button in the playback controls calls `classifyTeams(videoId)` (`POST /videos/{id}/classify-teams`). On mount, `getTeamClassifications(videoId)` (`GET /videos/{id}/classify-teams`) is called to restore any previously computed classifications.

### `handleClassifyTeams()`

Sets `isClassifying = true`, calls `classifyTeams`, stores the returned `classifications` and `summary` in state, then sets `isClassifying = false`. Errors are caught and stored in `classifyError`.

### `handleOverrideTeam(trackId, team)`

Calls `overrideTeamClassification(videoId, trackId, team)` (`PATCH /videos/{id}/classify-teams`) and updates `teamClassifications` with the returned classifications dict.

### Pitch Legend

Below the pitch canvas, a legend is shown:
- When `teamClassifications` is non-empty: three coloured dots labelled "Ellistown" (gold), "Opposition" (blue), "Unclassified" (grey).
- When empty: "Each player has a unique color based on their track ID".

### Team Classification Panel

An expandable `<details>` panel (open by default when data is present) appears below the player list. It shows:
- Summary stats: Ellistown count, opposition count, average confidence, HSV cluster separation, and a list of low-confidence track IDs (confidence < 0.6).
- Per-team groups (`ellistown`, `opposition`, `referee`, `ignore`), each showing track ID badges with a jersey-colour swatch, a confidence bar, and a team assignment dropdown (`<select>`). Changing the dropdown calls `handleOverrideTeam` immediately.

The jersey-colour swatch uses `hsvToCss(h, s, v)` from `lib/pitch.ts` to convert OpenCV HSV (H 0–179, S/V 0–255) to a CSS `rgb()` string.

---

## Analysis Trim Slider

A slider row below the main frame scrubber that controls which portion of the clip is used for pitch display and KPI computation.

### State split — why two variables?

The trim slider uses two separate state variables to avoid expensive re-renders on every drag tick:

- **`trimDragFrame`** — updates on every `onChange` event. Only the label text re-renders (cheap).
- **`trimEndFrame`** — the committed value that drives `analysisPositions`, `drawPitch`, playback stop, and the KPI `?end_frame` parameter. Only changes when the user explicitly clicks **"Apply trim"** or **"Reset"**.

Without this split, every drag tick would re-filter `playerPositions` (potentially thousands of objects) and trigger a full `drawPitch` canvas redraw, causing lag.

### UI

```
[Analysis trim end:] [────────────────●───] [frame 312 / 375 (12.5s)]  [Apply trim]  [Reset]
```

- **Slider**: `min=processedStartFrame`, `max=processedEndFrame`, `value=trimDragFrame`. Accent colour orange (`#ff9900`).
- **Label**: always has fixed `minWidth: 120` and `whiteSpace: nowrap` to prevent layout shifts as the number changes — layout shifts would cause the slider itself to jump while dragging.
- **Apply trim button**: orange background when `trimDragFrame !== trimEndFrame` (pending change), green with "✓ Trim applied" when they match (trim is active and up-to-date).
- **Reset button**: always visible; sets both `trimDragFrame` and `trimEndFrame` back to `processedEndFrame`.

### Effect on pipeline steps

| What | Affected by trim? |
|------|-------------------|
| Pitch canvas dot map | Yes — `analysisPositions` filters to `frame_idx <= trimEndFrame` |
| Player badges (frame) | Yes — same `analysisPositions` filter |
| Video playback | Yes — RAF stops at `trimEndFrame` |
| KPI computation | Yes — `?end_frame=trimEndFrame` sent to backend |
| Tracking / homography / annotations | No — trim is read-only |

---

## KPI Computation

### `handleComputeKpis()`

Calls `computeKpis(videoId, trimEndFrame)` (`POST /videos/{id}/compute-kpis?end_frame=N`). On success: stores the returned `KpiSummary` in `kpiSummary` state. On failure: stores error in `kpiError`.

The `trimEndFrame` parameter ensures the backend filters positions to `frame_idx <= trimEndFrame` before computing any metrics — matching what the frontend's `analysisPositions` shows.

### Clip Summary Panel

Displayed as a brief text summary above the detailed KPI panels. Reads from `kpiSummary` to produce:

- **Duration and frame count** from `clip_meta`.
- **Detected zone** (`detectedZone`) — inferred by counting how many player-frames fall in each pitch third (defensive: y 0–46.7 m; middle: 46.7–93.3 m; attacking: 93.3–140 m). The zone with the most player-frames is the detected zone.
- **Clip mode** (`clipMode`) — `'score'` (Ellistown attacking) or `'defense'` (Ellistown defending). Derived from `detectedZone`: `attacking` → `'score'`, `defensive` → `'defense'`, `middle` → determined by which team's centroid is closer to goal.
- **Top distances** — top 3 players by `total_distance_m`.
- **Depth sentence** — describes how the team centroids' relative depth (goal-side position) changed between the first and last frame where both teams are present in `spatial_timeseries`. Format: `"Clip start: [team] X.Xm goal-side · Clip end: [team] Y.Ym goal-side"`. Uses `oppGoalSide(eY, oY)` to determine which team is closer to goal given the `detectedZone` direction.

### Spatial KPIs Panel (`<details>`)

An expandable panel with:
- **Centroid separation** stats: mean / min / max from `spatial_summary.centroid_separation_m`.
- **Team centroids table**: per-team mean centroid (x, y in meters) and mean spread (m²). The team closer to goal is labelled "closer to goal".
- **Depth sentence**: same start→end comparison as the clip summary, shown again here for reference.
- **Zone balance** (if `zone_balance_timeseries` is present): for each team, a bar chart of % frames in defensive / middle / attacking third.

### Depth sentence logic

```typescript
const tsKeys = Object.keys(spatial_timeseries).map(Number).sort(...)
const bothPresent = tsKeys.filter(k => both teams have centroid_y_m at frame k)
const f0 = spatial_timeseries[bothPresent[0]]   // first frame both teams visible
const f1 = spatial_timeseries[bothPresent[last]] // last frame both teams visible
const oppGoalSide = (eY, oY) =>
  detectedZone === 'attacking' ? oY > eY : oY < eY
// If goal is at high-y (attacking zone): higher y = closer to goal
// If goal is at low-y (defensive zone): lower y = closer to goal
```

The sentence describes the *relative depth gap* between team centroids — not an absolute position. "Opposition 7.9m goal-side" means the opposition centroid was 7.9m closer to goal than Ellistown's centroid at that moment.

---

## Mapping View

An expandable `<details>` panel that shows the warped frame for the current frame index (`GET .../frames/{frame}/warped`). Indicates whether the current frame is an anchor frame or a propagated frame based on `homographyFrameIndices`.
