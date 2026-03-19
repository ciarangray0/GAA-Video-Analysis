# `ResultsViewer` Component

Displays the processed results: a side-by-side view of the original video and a 2D pitch canvas with player positions. Supports playback, frame stepping, speed control, and a BotSort overlay toggle.

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
| `isSyncMode` | When true, the video element is synced to `currentFrame` when not playing |
| `showBotSortOverlay` | Toggle for the BotSort bounding-box overlay image |
| `videoObjectUrl` | Blob URL created from `videoFile` |
| `showMappingView` | Whether the warped-frame debug panel is open |

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

## `getFramesWithPositions() → number[]`

Returns a sorted array of all frame indices that have at least one `PlayerPosition`. Used by `goToFrame` and `skipFrames` to navigate only to frames with data.

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
// Reset to start if ended or past the end
if (video.ended || video.currentTime >= processedEndFrame / fps) {
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
if (frameIdx > processedEndFrame) {
  video.pause(); setIsPlaying(false); return
}
onFrameChange(frameIdx)
animFrameRef.current = requestAnimationFrame(onPlaybackFrame)
```

Converts the video's `currentTime` to a frame index using `processedFps`. Stops playback at `processedEndFrame` (the last interpolated frame) even if the video continues beyond. Calls `onFrameChange` to update `currentFrame` in parent, which triggers the pitch canvas redraw.

---

## `stopPlayback()`

Cancels the RAF loop and pauses the video.

---

## Video–Pitch Sync Effect

```typescript
useEffect(() => {
  if (!isPlaying && isSyncMode && video.readyState >= 2) {
    const timeInSeconds = currentFrame / videoMetadata.fps
    if (Math.abs(video.currentTime - timeInSeconds) > 0.1) {
      video.currentTime = timeInSeconds
    }
  }
}, [currentFrame, isPlaying, isSyncMode, ...])
```

When not playing and sync mode is on, seeks the video to match `currentFrame`. The 0.1-second threshold prevents unnecessary seeks when the video is already close (small floating-point drift from the playback loop). Only runs when `readyState >= 2` (video has enough data to seek).

---

## `drawPitch` Effect

```typescript
useEffect(() => {
  if (canvasRef.current && playerPositions.length > 0) {
    drawPitch(canvasRef.current, playerPositions, currentFrame)
  }
}, [currentFrame, playerPositions])
```

Redraws the entire pitch canvas whenever `currentFrame` changes. `drawPitch` is a pure function — see `lib/OVERVIEW.md` for its implementation.

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

## Mapping View

An expandable `<details>` panel that shows the warped frame for the current frame index (`GET .../frames/{frame}/warped`). Indicates whether the current frame is an anchor frame or a propagated frame based on `homographyFrameIndices`.
