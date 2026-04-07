# `PipelineSteps` Component

This component renders the four sequential processing steps (A through D) of the analysis pipeline. Think of it as a control panel: each step has a "Run" button, shows its results when done, and visually indicates whether its results are still up-to-date ("STALE" badge) or currently running (spinner).

The four steps in order:
- **Step A** — Run YOLO object detection + BotSort tracking on the video
- **Step B** — Compute homographies (the maths that maps video pixels to pitch coordinates) using your annotations
- **Step C** — Map all detected players onto the pitch coordinate system
- **Step D** — Interpolate and smooth the player trajectories

---

## What is a component?

In React, a component is just a function that returns HTML-like code (called JSX). `PipelineSteps` is a component. It receives data from its parent via props and fires callback functions to tell the parent when something changes (e.g. a step finished running).

---

## Props — data passed in from the parent

Props are like arguments passed to the component. The parent component tells `PipelineSteps` about the current state of the world:

| Prop | What it is |
|------|-----------|
| `videoMetadata` | Info about the video — its ID (used in every API call), total frame count, fps |
| `anchorFrames` | The current list of annotations — used to build the request body for step B |
| `stepAResult` ... `stepDResult` | The results of each step, or `null` if not yet run |
| `staleSteps` | A `Set` of step IDs (e.g. `{"B", "C"}`) that are out of date — shown as "STALE" badges |
| `runningSteps` | A `Set` of step IDs currently executing — used to disable buttons |
| `onStepAComplete` ... `onStepDComplete` | Functions the parent gives us. We call these with results when a step finishes. This is how child components communicate results back to the parent. |
| `onStepsMarkedStale`, `onStepsClearedStale` | Functions to tell the parent to update the stale set |
| `onRunningStepChange` | Tells the parent `"add"` or `"remove"` a step from the running set |
| `onError`, `onStatusChange` | Functions to display messages in the parent's status bar |
| `logApiCall` | A function to append a message to the debug log in the sidebar |

---

## Local state

This component has a small amount of its own state — data that only it needs to know about:

```typescript
const [anchorQuality, setAnchorQuality] = useState(null)
const [stepBVersion, setStepBVersion] = useState(0)
const [sgLongWindow, setSgLongWindow] = useState(31)
const [sgMidWindow, setSgMidWindow] = useState(11)
const [maxVelPx, setMaxVelPx] = useState(50)
```

| State | What it is |
|-------|-----------|
| `anchorQuality` | The per-anchor reprojection quality report fetched after step B runs |
| `anchorQualityError` | An error message if the quality fetch failed |
| `stepBVersion` | A counter that increments every time step B runs. Used to force the browser to reload warped-frame thumbnails (see "cache busting" below) |
| `sgLongWindow`, `sgMidWindow`, `maxVelPx` | Smoothing parameters for step D. Shown as editable number inputs so the user can tune them before running. |

---

## `apiFetch` — a logging wrapper around `fetch`

```typescript
const apiFetch = async (url: string, options?: RequestInit) => {
  logApiCall(`→ ${method} ${url}`)
  const response = await fetch(url, options)
  logApiCall(`← ${response.status} (${elapsed}ms)`)
  return response
}
```

`fetch` is the browser's built-in function for making HTTP requests. `apiFetch` is a thin wrapper that adds logging: before the request it logs "→ POST /track", and after it logs "← 200 (1234ms)". This populates the debug log in the sidebar so you can see what API calls were made and how fast they responded.

Steps A, C, and D use `apiFetch`. Step B calls `computeHomographies()` from `lib/api.ts` directly and logs manually — this is because step B's API function has its own structured error handling that would be awkward to route through the generic wrapper.

---

## Step A — Tracking (`runStepA`)

This step tells the backend to run YOLO object detection and BotSort player tracking on the video.

**What happens:**
1. `POST /videos/{id}/track` — kicks off tracking. This can take a while (YOLO processes every frame).
2. `GET /videos/{id}/detections` — fetches detection counts so we can show "N detections, M unique tracks" in the results panel.
3. Calls `onStepAComplete({frames_processed, tracks, num_detections})` to pass results to the parent.
4. Marks steps B, C, D as stale — tracking outputs are the foundation for everything else, so all downstream results are now out of date.

**Results display:** video ID, fps, total frames, number of detections, number of unique tracked players.

---

## Step B — Homographies (`runStepB`)

This step takes your annotations (frame pixel → pitch coordinate pairs) and computes a homography matrix for each annotated frame. It also propagates those homographies to un-annotated frames using optical flow.

**Annotation filtering:**

Before sending anything to the server, we filter out frames that are not useful:

```typescript
const validAnnotations = anchorFrames
  .filter(af => !af.isSkipped && (af.points.length > 0 || (af.lines || []).length > 0))
  .map(af => ({ frame_idx: af.frame_idx, points: af.points, lines: af.lines || [] }))
```

A frame is included if it is NOT skipped AND has at least one point OR at least one line annotation. Frames that are skipped, or that have no annotations at all, are excluded. The server receives only the useful data.

**What happens:**
1. `POST /videos/{id}/homographies/v3` with the filtered annotation list.
2. Builds a `StepBResult` object from the response.
3. Calls `onStepBComplete(result, data.frames)` to pass results up.
4. Increments `stepBVersion` to bust thumbnail caches (see below).
5. Marks steps C and D as stale (new homographies → player positions need to be recomputed).
6. Immediately fetches `GET /videos/{id}/homographies/anchor-quality` to get the reprojection error report.

**Cache busting with `stepBVersion`:**

The results panel shows warped-frame thumbnails — images fetched from the server like `GET /videos/{id}/frames/30/warped`. The browser caches these images by URL. If step B runs again with different annotations, the URL is the same but the image is different. The browser would show the old cached version.

The fix: append `?v={stepBVersion}` to the thumbnail URLs. Every time step B runs, `stepBVersion` increments (0 → 1 → 2...). The URL changes, so the browser treats it as a new image and re-fetches.

**Results display:**
- How many anchor frames were computed, how many per-frame homographies were propagated.
- A list of any "bad quality" frames (shown in red).
- A table showing per-anchor details: keypoint count, line count, whether the solver converged, reprojection error (mean + max), and any warnings.
- An expandable section with per-keypoint error breakdowns.
- Three thumbnail images per anchor: the original frame, the warped frame (frame mapped onto the pitch), and the warped frame with player positions overlaid.

---

## Step C — Player Mapping (`runStepC`)

This step takes the per-frame homographies from step B and uses them to map every detected player's foot position into pitch coordinates.

**What happens:**
1. `mapPlayers(videoId)` from `lib/api.ts` — calls `POST /videos/{id}/map_players`.
2. Calls `onStepCComplete({positions, total})`.
3. Marks step D as stale (new positions → interpolation needs to rerun).

**Results display:** total number of positions mapped, plus an expandable table showing the first 20 positions (frame index, track ID, x/y in pitch metres, source method).

---

## Step D — Interpolation (`runStepD`)

Player tracking has gaps — a player might disappear behind another player for a few frames, or the detector missed a frame. Step D fills in these gaps using linear interpolation and then smooths the result using a Savitzky-Golay filter (a mathematical smoothing algorithm that reduces jitter without distorting the overall motion).

**User-tunable parameters:**

Before running, the user can adjust three parameters shown as number inputs:

| Parameter | What it controls |
|-----------|-----------------|
| `sgLongWindow` | Window size for the "long" Savitzky-Golay smoothing pass (larger = smoother but more lag) |
| `sgMidWindow` | Window size for the "medium" pass |
| `maxVelPx` | Maximum allowed velocity in pitch pixels per frame. Positions that jump further than this between frames are clamped (prevents tracking glitches from creating huge teleportation jumps). |

**What happens:**
1. `interpolateTrajectories(videoId, 0, num_frames-1, {sgLongWindow, sgMidWindow, maxVelPx})`.
2. `getPlayerPositions(videoId)` — fetches the complete final position list.
3. `onStepDComplete(result, allPositions, startFrame, endFrame, fps)` — passes everything to the parent, which stores it and shows `ResultsViewer`.
4. `onStatusChange('Pipeline complete!')` — updates the parent's status bar.

**Results display:** how many interpolated frames were generated and which interpolation method was used.

---

## `validAnnotationCount` — computed with useMemo

```typescript
const validAnnotationCount = useMemo(
  () => anchorFrames.filter(af =>
    !af.isSkipped && (af.points.length > 0 || (af.lines || []).length > 0)
  ).length,
  [anchorFrames]
)
```

`useMemo` is a React hook that caches the result of a computation and only recalculates it when its dependencies change. Here, we count how many annotation frames are usable (not skipped, has points or lines). This number is recalculated only when `anchorFrames` changes — not on every render.

The step B "Run" button is disabled when `validAnnotationCount === 0`. Without any valid annotations, the server has nothing to compute a homography from.

---

## Stale logic — why results go out of date

The steps depend on each other in a chain: A → B → C → D. If you re-run an earlier step, later steps' results are now based on old data.

- Completing A marks B, C, D stale (new tracking → everything downstream needs rerunning)
- Completing B marks C, D stale
- Completing C marks D stale
- Completing D clears D stale

When a step is stale, its panel shows a "STALE" badge next to the results, warning the user that those numbers may not reflect the current state of the pipeline.

---

## Display formatting helpers

All the colour-coded labels and badges come from `utils/formatters.ts`. These are pure functions — they take a value and return a string or colour. They are kept separate to avoid cluttering the component logic.

| Function | What it produces |
|----------|-----------------|
| `reprErrorLabel(val)` | A label like `"15px ⚠"` with an appropriate icon |
| `reprErrorColor(val)` | A CSS colour string — green if error is small, amber if medium, red if large |
| `qualityBadge(q)` | `"✅ good"`, `"⚠️ warning"`, or `"❌ bad"` |
| `qualityColor(q)` | CSS colour matching the quality level |
| `verdictBadge(v)` | `"✓"`, `"⚠"`, or `"✗"` for a per-keypoint verdict |
| `impactColor(impact)` | CSS colour for "helpful" / "marginal" / "harmful" impact ratings |
