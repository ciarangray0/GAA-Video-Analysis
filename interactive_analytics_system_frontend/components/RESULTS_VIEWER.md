# `ResultsViewer` Component

This is the final output screen — shown after the full pipeline has run. It displays the original video on the left and a 2D bird's-eye pitch map on the right. As the video plays, coloured dots on the pitch map move to show where each player is at that moment. It also handles team classification (figuring out which player is on which team) and KPI (Key Performance Indicator) computation.

---

## What is a component?

In React, a component is just a function that returns HTML-like code (called JSX). Every time its data changes, React re-runs the function and updates what you see on screen. `ResultsViewer` is a large component that manages video playback, canvas drawing, and several API calls.

---

## Props — data passed in from the parent

| Prop | What it is |
|------|-----------|
| `videoMetadata` | fps and total frame count — used to convert between frame numbers and timestamps |
| `videoFile` | The raw video file the user uploaded — used to play the video locally without re-downloading it |
| `playerPositions` | Every player position from the interpolation step — thousands of `{frame_idx, track_id, x_pitch, y_pitch}` objects |
| `currentFrame` | Which frame is currently shown (controlled by the parent so other components can sync to it) |
| `onFrameChange` | A function the parent gives us. We call it whenever the frame changes (playback, scrubbing, step buttons). |
| `processedStartFrame`, `processedEndFrame` | The range of frames that were processed — playback stays within this range |
| `homographyFrameIndices` | Which frames are anchor frames — shown in the debug mapping panel |
| `processedFps` | The fps value used during processing |

---

## Local state — what this component tracks itself

```typescript
const [isPlaying, setIsPlaying] = useState(false)
const [playbackSpeed, setPlaybackSpeed] = useState(1)
const [showBotSortOverlay, setShowBotSortOverlay] = useState(false)
```

Think of `useState` like a variable with a built-in alarm. When you call the setter function (e.g. `setIsPlaying(true)`), React notices the change and re-renders the component. You must never write `isPlaying = true` directly.

| State | Starts as | What it tracks |
|-------|-----------|----------------|
| `isPlaying` | `false` | Whether the video + pitch animation is currently playing |
| `playbackSpeed` | `1` | Playback speed multiplier (0.25×, 0.5×, 1×, 2×, 4×) |
| `showBotSortOverlay` | `false` | Whether to show the BotSort bounding-box overlay on the video |
| `videoObjectUrl` | `null` | A temporary browser URL created from the video file (explained below) |
| `showMappingView` | `false` | Whether the warped-frame debug panel is expanded |
| `teamClassifications` | `{}` | A dictionary mapping `track_id` → `{team, confidence, mean_hsv}`. Empty until "Classify Teams" is run. |
| `classifySummary` | `null` | Summary statistics from the most recent team classification |
| `isClassifying` | `false` | True while the classify-teams API call is in progress |
| `classifyError` | `null` | An error message if classification failed |
| `kpiSummary` | `null` | The KPI results returned by the backend, or `null` until computed |
| `isComputingKpis` | `false` | True while the compute-kpis API call is in progress |
| `kpiError` | `null` | An error message if KPI computation failed |
| `trimEndFrame` | `processedEndFrame` | The **committed** end frame for analysis (see trim slider section) |
| `trimDragFrame` | `processedEndFrame` | The **live** slider position while dragging (see trim slider section) |

---

## useRef — values that don't trigger re-renders

```typescript
const canvasRef    = useRef<HTMLCanvasElement>(null)
const videoPlayerRef = useRef<HTMLVideoElement>(null)
const animFrameRef = useRef<number | null>(null)
```

A `useRef` stores a value that you can read or write without causing the component to re-render. The main uses here:
- `canvasRef` and `videoPlayerRef` give direct access to the `<canvas>` and `<video>` HTML elements so we can call their methods (e.g. `canvas.getContext('2d')`, `video.play()`).
- `animFrameRef` stores the ID of the current animation frame request, so we can cancel it when playback stops.

---

## Creating a video URL from the file: the blob URL

```typescript
useEffect(() => {
  const url = URL.createObjectURL(videoFile)
  setVideoObjectUrl(url)
  return () => URL.revokeObjectURL(url)
}, [videoFile])
```

The user has already uploaded the video to the server. But to play it locally in the browser's `<video>` element, we need a URL the browser can use. `URL.createObjectURL(videoFile)` creates a temporary "blob URL" (looks like `blob:http://localhost/abc-123`) that points directly to the file in memory. This means:
- No second HTTP upload — the video plays from the file the user already selected.
- It works offline after the initial upload.

The `return () => URL.revokeObjectURL(url)` part runs when the component unmounts (disappears from the page). It frees the memory. This pattern — a function returned from `useEffect` — is called a "cleanup function".

**useEffect** is a React hook that runs code in response to changes. The `[videoFile]` at the end is the "dependency list". This effect runs once when `videoFile` is first set, and again only if `videoFile` changes. Without this list, the effect would run on every single render, creating a new blob URL each time.

---

## `analysisPositions` — filtering with useMemo

```typescript
const analysisPositions = useMemo(
  () => playerPositions.filter(p => p.frame_idx <= trimEndFrame),
  [playerPositions, trimEndFrame]
)
```

`useMemo` caches the result of a calculation and only recomputes it when its dependencies change. Here, we filter the full player positions list down to only frames within the trim end point. This filtered list is used everywhere: canvas drawing, player count badges, playback stop logic. The raw `playerPositions` prop is never modified — trimming is purely a read-time filter.

**Why useMemo?** `playerPositions` can contain thousands of objects. Running `.filter()` on every render (which could happen 60 times per second during playback) would be slow. `useMemo` means the filter only runs when `playerPositions` or `trimEndFrame` actually changes.

---

## `goToFrame(frameIdx)` — snapping to valid frames

When the user drags the scrubber to frame 47, there might not be any player data for frame 47 (tracking has gaps). `goToFrame` snaps to the nearest frame that DOES have player data:

```typescript
let nearest = frames[0]
let minDist = Math.abs(frameIdx - nearest)
for (const f of frames) {
  const dist = Math.abs(frameIdx - f)
  if (dist < minDist) { minDist = dist; nearest = f }
}
onFrameChange(nearest)
```

It loops through every frame that has at least one player position and finds whichever one is closest to the requested frame. Then it tells the parent (via `onFrameChange`) to use that frame.

---

## Playback — how video and pitch canvas stay in sync

Playback is more complex than just calling `video.play()`. The pitch canvas also needs to update in sync with the video. Here is how it works:

### `startPlayback()`

```typescript
video.playbackRate = playbackSpeed
if (video.ended || video.currentTime >= trimEndFrame / fps) {
  video.currentTime = startTime   // rewind if at the end
}
setIsPlaying(true)
video.play()
  .then(() => {
    animFrameRef.current = requestAnimationFrame(onPlaybackFrame)
  })
  .catch(err => {
    setIsPlaying(false)
  })
```

`video.play()` returns a **Promise** — it doesn't play immediately, it schedules playback and tells you when it actually starts (via `.then()`). The RAF (RequestAnimationFrame) loop only starts inside `.then()`. This is important: if you started the RAF loop before the video was actually playing, the pitch canvas would update but the video frame wouldn't move, causing the two to get out of sync.

`video.play()` can also fail (browsers block autoplay on some pages). The `.catch()` handles this gracefully by setting `isPlaying` back to false.

### `onPlaybackFrame()` — the animation loop

`requestAnimationFrame` is a browser API that calls your function approximately 60 times per second, in sync with the screen's refresh rate. It is used here to update the pitch canvas smoothly during playback.

```typescript
const frameIdx = Math.round(video.currentTime * fps)
if (frameIdx > trimEndFrame) {
  video.pause()
  setIsPlaying(false)
  return
}
onFrameChange(frameIdx)
animFrameRef.current = requestAnimationFrame(onPlaybackFrame)
```

Each call:
1. Converts the video's current time (in seconds) to a frame index by multiplying by fps.
2. Checks if we've hit the trim end — if so, stops.
3. Calls `onFrameChange(frameIdx)` to tell the parent the current frame. The parent updates `currentFrame`, which triggers the canvas redraw effect.
4. Schedules the next call with `requestAnimationFrame`.

Notice it always calls `requestAnimationFrame` at the END to schedule the next tick — this creates a loop that continues until playback is stopped.

### `stopPlayback()`

Cancels the RAF loop (using the ID stored in `animFrameRef`) and calls `video.pause()`.

### Video-pitch sync when NOT playing

```typescript
useEffect(() => {
  if (!isPlaying && video.readyState >= 2) {
    const timeInSeconds = currentFrame / videoMetadata.fps
    if (Math.abs(video.currentTime - timeInSeconds) > 0.1) {
      video.currentTime = timeInSeconds
    }
  }
}, [currentFrame, isPlaying])
```

When the user scrubs the frame slider without playing, we need to keep the video's current position in sync with `currentFrame`. This effect watches `currentFrame` and seeks the video whenever it changes. The 0.1-second threshold prevents unnecessary seeks for tiny floating-point differences. `readyState >= 2` means the video has loaded enough data to actually seek.

---

## Drawing the pitch canvas: the `drawPitch` effect

```typescript
useEffect(() => {
  if (canvasRef.current && analysisPositions.length > 0) {
    drawPitch(canvasRef.current, analysisPositions, currentFrame, teamClassifications, showTrails)
  }
}, [currentFrame, analysisPositions, teamClassifications, showTrails])
```

Every time `currentFrame`, `analysisPositions`, or `teamClassifications` changes, this effect redraws the entire pitch canvas. The `drawPitch` function (in `lib/pitch.ts`) handles:
- Drawing the pitch lines (goals, 20m lines, 45m lines, etc.)
- Drawing a coloured dot for each player at their position in `currentFrame`
- If `teamClassifications` is provided: colouring dots by team (yellow for Ellistown, blue for opposition)
- If `showTrails` is on: drawing a trail of recent positions behind each player dot

**Why redraw the whole canvas?** The Canvas 2D API does not track individual drawn shapes — once something is drawn, it is pixels. To move a dot, you must erase everything and redraw from scratch. This is standard practice.

---

## The BotSort overlay

```typescript
{showBotSortOverlay && (
  <img
    src={`/videos/${videoId}/frames/${currentFrame}/detections_overlay`}
    key={currentFrame}
    onError={() => { /* hide and show fallback */ }}
  />
)}
```

When toggled on, this renders an `<img>` element that fetches a server-generated overlay image showing the bounding boxes from BotSort tracking. The `key={currentFrame}` prop is important: React uses `key` to decide whether to reuse an element or create a new one. When `currentFrame` changes, `key` changes, React creates a new `<img>` element with the new `src`, forcing it to load the correct frame's overlay rather than showing the cached old one.

---

## The trim slider — two-state design

The trim slider lets you exclude the end of a clip from analysis. For example, if a clip is 400 frames long but the play ends at frame 312, you can trim to frame 312 so KPIs and pitch rendering ignore the trailing frames where players are standing around.

### Why two separate state variables?

```typescript
const [trimEndFrame, setTrimEndFrame] = useState(processedEndFrame)   // committed
const [trimDragFrame, setTrimDragFrame] = useState(processedEndFrame) // live while dragging
```

As the user drags the slider, `onChange` fires dozens of times per second. If each drag tick updated `trimEndFrame` (the committed value), it would re-filter thousands of player positions AND redraw the pitch canvas on every tick — causing noticeable lag.

The two-state solution:
- **`trimDragFrame`** updates on every drag tick. Only the text label (`"frame 312 / 375 (12.5s)"`) re-renders — this is cheap.
- **`trimEndFrame`** only changes when the user clicks **"Apply trim"**. That is when the expensive re-filter and canvas redraw happen.

This is a common React performance pattern: separate "live" UI state from "committed" data state.

### Apply trim button appearance

- Orange background when `trimDragFrame !== trimEndFrame` — signals a pending change.
- Green with "✓ Trim applied" when they match — confirms the trim is active.

### What the trim affects

| What | Affected? |
|------|-----------|
| Pitch canvas dot map | Yes — only frames up to `trimEndFrame` are drawn |
| Player badges | Yes — same filter |
| Video playback | Yes — RAF stops at `trimEndFrame` |
| KPI computation | Yes — `?end_frame=trimEndFrame` sent to the backend |
| Tracking / homographies / annotations | No — trim is purely a read-time filter |

---

## Team classification

The system can automatically classify players by team using their jersey colours (analysed in HSV colour space by the backend).

### `handleClassifyTeams()`

Calls `POST /videos/{id}/classify-teams`. On success, stores:
- `teamClassifications` — a dictionary mapping `track_id` to `{team, confidence, mean_hsv}`
- `classifySummary` — aggregate statistics (cluster separation, low-confidence players, etc.)

On mount, `GET /videos/{id}/classify-teams` is called to restore any previously computed classifications — so refreshing the page doesn't lose your classification.

### `handleOverrideTeam(trackId, team)`

If the automatic classification is wrong for a player, the user can manually correct it via a dropdown. This calls `PATCH /videos/{id}/classify-teams` and updates `teamClassifications` with the response.

### How team colours affect the pitch canvas

The `teamClassifications` dictionary is passed to `drawPitch`. When non-empty, player dots are coloured:
- Gold/yellow = Ellistown
- Blue = opposition
- Tracks labelled `'referee'` or `'ignore'` are hidden

When `teamClassifications` is empty (not yet run), each player gets a unique colour based on their track ID.

---

## KPI Computation

### `handleComputeKpis()`

Calls `POST /videos/{id}/compute-kpis?end_frame={trimEndFrame}`. The `trimEndFrame` parameter ensures the backend computes KPIs over the same frame range that the frontend is displaying — they will always match.

On success, `kpiSummary` is set and the KPI panels are shown.

### Clip Summary Panel

A brief plain-English summary is generated from `kpiSummary`:

- **Duration** and frame count from `clip_meta`.
- **Detected zone** — computed by counting how many player-frames fall in each third of the pitch (defensive: 0–46.7m, middle: 46.7–93.3m, attacking: 93.3–140m). The zone with the most player-frames is the "detected zone". This tells you which part of the pitch the clip is from.
- **Clip mode** — `'score'` (Ellistown attacking toward goal) or `'defense'` (Ellistown defending). Derived from which zone was detected and where each team's centroid is relative to goal.
- **Top distances** — the three players who covered the most ground.
- **Depth sentence** — a plain-English description of how the relative depth gap between team centroids changed from the start to the end of the clip. For example: "Clip start: Opposition 7.9m goal-side · Clip end: Ellistown 3.2m goal-side".

### Spatial KPIs Panel

An expandable section showing:
- **Centroid separation** — the mean/min/max distance between the two teams' centroids over the clip.
- **Team centroids table** — where each team's centroid was on average (in metres), and how spread out (compact) they were.
- **Zone balance** — for each team, what percentage of frames they spent in each third of the pitch (shown as a bar chart).

### The depth sentence logic

```typescript
const oppGoalSide = (eY, oY) =>
  detectedZone === 'attacking' ? oY > eY : oY < eY
```

This function determines which team is "closer to goal" given the teams' centroid Y positions. The direction depends on the detected zone:
- In the **attacking** zone, goal is at the far (high-Y) end, so higher Y = closer to goal.
- In the **defensive** zone, goal is at the near (low-Y) end, so lower Y = closer to goal.

The depth sentence shows the *relative gap* — not absolute positions. "Opposition 7.9m goal-side" means the opposition centroid was 7.9m closer to goal than Ellistown's centroid at that moment.

---

## Mapping view (debug panel)

An expandable `<details>` section that shows the warped frame for the current frame index (`GET .../frames/{frame}/warped`). It also labels whether the current frame is an anchor frame (directly annotated) or a propagated frame (computed from a nearby anchor via optical flow). This is mainly useful during development to visually verify that the homography is correct for a given frame.

---

## Debug coordinate table

Always visible below the main view. For the current frame, it lists every player's pitch coordinates, their converted display coordinates, and whether they are within bounds:

```typescript
const isOutOfBounds = pos.x_pitch < 0 || pos.x_pitch > PITCH_CANVAS_W || ...
```

Out-of-bounds rows are highlighted in red. This table was essential during development to catch homography bugs — if a player's pitch coordinates are -200 or 3000, something in the pipeline went wrong.
