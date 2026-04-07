# Components Overview

This document explains what each of the five UI components does and how it is wired into the rest of the app. If you are new to React, read the OVERVIEW.md in the root first — it explains state, props, and callbacks.

---

## How components communicate

In React, data flows **down** (from parent to child via props) and events flow **up** (from child to parent via callback functions). Think of it like a manager and employees: the manager (`index.tsx`) holds all the important information and decisions, but the employees (components) do the actual visible work and report back when something happens.

```
index.tsx  →  passes data as props  →  VideoUploader
           ←  receives callbacks   ←  VideoUploader
```

No component directly talks to another sibling. If `VideoUploader` needs to give data to `PipelineSteps`, it goes up to `index.tsx` first and then back down.

---

## `VideoUploader`

**Role**: The first thing the user interacts with. Lets them pick a video file and upload it to the backend.

**What it renders**: A file `<input>` element and an "Upload" button.

**What it does**:
1. When the user selects a file, it stores that file in local state.
2. When the user clicks "Upload", it calls `uploadVideo(file)` from `lib/api.ts`. This sends the file to the backend via an HTTP POST request.
3. When the backend responds with video metadata (fps, frame count, duration, etc.), it calls `onUploadSuccess(metadata, file)` to give that data back to `index.tsx`.
4. If something goes wrong (e.g. wrong file format, backend down), it shows an error message inline.

**Props in**: none
**Callbacks out**: `onUploadSuccess(metadata: VideoMetadata, file: File)`

---

## `AnchorFrameAnnotator`

**Role**: The most complex component. Lets the user tell the system "this point in the camera image corresponds to this point on the real pitch." This is the human annotation step that makes the bird's-eye-view transformation possible.

Full details are in `ANCHOR_FRAME_ANNOTATOR.md`. Here is a high-level summary.

### What it renders

Two canvas elements side by side:
- **Left canvas**: shows the video frame (a single still image). The user clicks on this to mark where a pitch landmark appears in the camera image.
- **Right canvas**: shows a 2D top-down pitch diagram. The user clicks on this to say where that landmark sits in real-world pitch coordinates.

### Two annotation modes

**Point mode**: The user clicks a named vertex (like "45m_line_left_post") on the pitch diagram, then clicks the matching spot in the video frame. This records an exact landmark.

**Line mode**: The user selects a pitch line (like "45m line"), clicks two points along that line in the video frame, then confirms. The system uses both points to constrain the homography calculation.

### Auto-save

Every time the annotations change, a `useEffect` writes them to `localStorage` automatically. The user never needs to manually save.

**Props in**: `videoMetadata`, `anchorFrames`, `currentAnchorIdx`
**Callbacks out**: `onAnchorFramesChange(frames)`, `onCurrentIdxChange(idx)`

---

## `PipelineSteps`

**Role**: Runs the four backend processing steps and shows their results.

**What it renders**: Four collapsible panels labelled A, B, C, D. Each has a "Run" button, a result summary, and sometimes a detailed table.

### The four steps

**Step A — Tracking**: Calls `trackVideo(videoId)`. The backend runs YOLO object detection on every frame and BotSort tracking to link detections across frames into continuous player tracks. Returns the number of tracks and frames processed.

**Step B — Homography**: Calls `computeHomographies(videoId, annotations)`. Sends all the anchor frame annotations to the backend. The backend computes a perspective transform matrix (homography) for each anchor frame, then uses optical flow to propagate those matrices to every frame in between. Also fetches the anchor quality report and displays it as a colour-coded table so the user can see if any annotations were imprecise.

**Step C — Player Mapping**: Calls `mapPlayers(videoId)`. Takes every tracked player detection and projects its position from camera pixels into real-world pitch metres, using the per-frame homography matrices computed in step B. Returns all the 2D pitch positions.

**Step D — Interpolation**: Calls `interpolateTrajectories(videoId, start, end, params)`. Fills in frames where a player was not detected (e.g. behind another player or temporarily outside frame) using smooth interpolation. Then fetches the full position set with `getPlayerPositions(videoId)` and passes it back to `index.tsx` via `onStepDComplete`.

### Stale steps

When a step completes, it calls `onStepsMarkedStale` with the IDs of all downstream steps. For example, completing step A marks B, C, and D as stale (because they depend on A's output). This prompts the user to re-run those steps.

### Local state

`PipelineSteps` owns two pieces of local state that nothing else needs:

- `anchorQuality`: the reprojection quality report from step B, used only to render the quality table inside step B's panel.
- `stepBVersion`: a counter that increments each time step B runs, used to force a fresh quality fetch even if the video ID hasn't changed.

**Props in**: `videoMetadata`, `anchorFrames`, step results, `staleSteps`, `runningSteps`
**Callbacks out**: `onStepAComplete`, `onStepBComplete`, `onStepCComplete`, `onStepDComplete`, stale/running change callbacks

---

## `ResultsViewer`

**Role**: Shows the original video and the 2D pitch view side by side, in sync. Also handles team classification, KPI computation, and all the playback controls.

Full details are in `RESULTS_VIEWER.md`. Here is a summary.

### What it renders

- The original video in an HTML `<video>` element (left side).
- A pitch canvas (right side) that shows player dots at the current frame, updated in real time during playback.
- Playback controls: play/pause, speed multiplier, frame-step buttons.
- A BotSort overlay toggle (shows the detection bounding boxes from the original tracker overlaid on the video).
- A team classification panel with jersey colour swatches and dropdowns to override individual player assignments.
- KPI display: zone balance charts, team compactness, centroid depth summary.
- An "analysis trim" slider to exclude trailing dead-ball frames from KPI computation.

### How playback works

Playback is driven by `requestAnimationFrame` (RAF) — a browser API that calls your function roughly 60 times per second, in sync with the screen refresh rate. Think of it like a loop that runs every frame of animation.

Each RAF tick:
1. Checks whether it is time to advance to the next frame based on elapsed time and the selected playback speed.
2. If so, increments the current frame number and updates `currentFrame` state.
3. Seeks the HTML video element to the new timestamp: `video.currentTime = frame / fps`.
4. Redraws the pitch canvas using `drawPitch(canvas, positions, frame, teamClassifications)`.

This approach keeps the video and the pitch canvas in sync without depending on the video's own playback events, which can be unreliable when seeking frequently.

### How the pitch canvas is redrawn

The pitch canvas is a plain HTML `<canvas>` element. React does not manage what is drawn on it — the app calls `drawPitch()` directly using the canvas's 2D context. This is an intentional design choice: React is great for managing UI state, but imperative canvas drawing code fits awkwardly into React's declarative model. Keeping the drawing logic in `lib/pitch.ts` and calling it from a `useEffect` or RAF callback is a clean separation.

**Props in**: `videoMetadata`, `videoFile`, `playerPositions`, `currentFrame`, frame range, `homographyFrameIndices`
**Callbacks out**: `onFrameChange(frame)`

---

## `DebugLog`

**Role**: A scrollable sidebar that shows a timestamped log of every API call made during the session, plus a summary table of pipeline step results.

**What it renders**:
- A list of log entries, each showing: a direction arrow (request vs. response), the endpoint URL, the HTTP status code, and the time taken in milliseconds.
- A pipeline summary table listing what each step returned (track count, frame count, etc.).
- A "Clear" button.

**Why it exists**: When something goes wrong in the pipeline, the debug log lets you see exactly which backend call failed, what the status code was, and how long it took. This makes diagnosing problems much faster than opening browser DevTools every time.

**Props in**: `entries` (the log array), step results from A/B/C/D
**Callbacks out**: `onClear()`
