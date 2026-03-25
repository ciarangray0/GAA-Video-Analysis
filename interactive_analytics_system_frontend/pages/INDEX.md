# `pages/index.tsx`

Root page of the application. Owns all cross-step state and wires the five major UI components together.

---

## Key State Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `videoFile` | `null` | Raw `File` object — passed to `ResultsViewer` to create a blob URL for the HTML `<video>` element. Not uploaded via this state; `VideoUploader` handles the HTTP POST. |
| `videoMetadata` | `null` | Set by `VideoUploader.onUploadSuccess`. Contains `video_id` used in all subsequent API calls. |
| `anchorInterval` | `1` | Seconds. Controls how many frames `generateAnchorFrames` creates. |
| `anchorFrames` | `[]` | Array of `AnchorFrame` — the core annotation data structure. Each element has `frame_idx`, `isSkipped`, `points`, `lines`. |
| `currentAnchorIdx` | `0` | Index into `anchorFrames` — which anchor the annotator is currently showing. |
| `stepAResult` | `null` | Set after tracking completes. Holds `{frames_processed, tracks, num_detections}`. |
| `stepBResult` | `null` | Set after homography computation. Holds `{frames, per_frame_count, info}`. |
| `stepCResult` | `null` | Set after player mapping. Holds `{positions, total}`. |
| `stepDResult` | `null` | Set after interpolation. Holds `{frames_generated, method}`. |
| `staleSteps` | `new Set()` | Step IDs (`"A"`, `"B"`, `"C"`, `"D"`) that are out of date with current annotations. Displayed as "STALE" badges in PipelineSteps. |
| `runningSteps` | `new Set()` | Steps currently executing. Used to disable buttons. |
| `stepDoneRef` | `{B:false, C:false, D:false}` | `useRef` tracking whether downstream steps have results. Used to avoid false stale-marking on first annotation. |
| `playerPositions` | `[]` | Dense list of all `PlayerPosition` objects after interpolation. Passed to ResultsViewer. |
| `currentFrame` | `0` | Currently selected frame in ResultsViewer. Initialised to the first frame with player positions after step D completes. |
| `processedStartFrame` | `0` | Start frame passed to the interpolation step. |
| `processedEndFrame` | `0` | End frame passed to the interpolation step. |
| `homographyFrameIndices` | `[]` | Anchor frame indices returned from step B. Displayed in ResultsViewer to show which frames are anchor frames. |
| `processedFps` | `25` | FPS from `videoMetadata.fps`. Used for video/pitch sync. |
| `status`, `error` | `''` | User-facing status/error messages shown in the status bar. |
| `debugLog` | `useRef([])` | Accumulates API call log strings. `useRef` prevents rerenders on every log append. |
| `debugLogEntries` | `[]` | Synced copy of `debugLog.current` used as prop for `DebugLog`. |

---

## `generateAnchorFrames()`

Triggered by the "Generate Anchor Frames" button (step 2).

```typescript
for (let seconds = 0; seconds <= duration_seconds; seconds += anchorInterval) {
  const frameIdx = Math.floor(seconds * fps)
  frames.push({ frame_idx: frameIdx, isSkipped: false, points: [], lines: [] })
}
```

After generating, checks `localStorage` for `"gaa_annotations_{videoFile.name}"`. If saved data exists, shows a confirm dialog. On confirmation, merges: for each generated frame, if a saved frame with the same `frame_idx` exists, replaces `isSkipped`, `points`, `lines` from the saved version.

---

## `handleAnchorFramesChange(frames)`

Called by `AnchorFrameAnnotator` whenever annotations are modified.

1. Updates `anchorFrames` state.
2. Reads `stepDoneRef` to see which downstream steps have results.
3. Marks those steps stale (avoids marking stale before any step has run — `stepDoneRef` uses `useRef` so it doesn't cause a render cycle itself).

---

## `markStale(steps)` / `clearStale(steps)`

```typescript
const markStale = (steps: string[]) => {
  setStaleSteps(prev => { const next = new Set(prev); steps.forEach(s => next.add(s)); return next })
}
const clearStale = (steps: string[]) => {
  setStaleSteps(prev => { const next = new Set(prev); steps.forEach(s => next.delete(s)); return next })
}
```

Both use the functional updater form to avoid stale closure bugs. PipelineSteps calls these via `onStepsMarkedStale` and `onStepsClearedStale` props.

---

## `logApiCall(entry)`

```typescript
const logApiCall = (entry: string) => {
  debugLog.current = [...debugLog.current, entry]
  setDebugLogEntries([...debugLog.current])
}
```

Spreads the array on every update to ensure React sees a new reference for the state update.

---

## Step D Completion Callback

```typescript
onStepDComplete={(result, positions, start, end, fps) => {
  setStepDResult(result)
  setPlayerPositions(positions)
  setProcessedStartFrame(start)
  setProcessedEndFrame(end)
  setProcessedFps(fps)
  const firstFrame = positions.length > 0 ? Math.min(...positions.map(p => p.frame_idx)) : start
  setCurrentFrame(firstFrame)
}}
```

Sets `currentFrame` to the first frame that has player positions, so ResultsViewer starts at a frame with something to show rather than frame 0 which may be outside the processed range.

---

## `hasResults` and the Full-Bleed Layout

```typescript
const hasResults = playerPositions.length > 0 && videoMetadata !== null && videoFile !== null
```

When `hasResults` is true the page switches from the normal scrolling pipeline layout to a **full-viewport two-column layout**:

```
┌─────────────────────┬──────────────────────────────────┐
│  .pipeline-panel    │  .results-panel                  │
│  (380 px, fixed)    │  (flex: 1, scrollable)           │
│  pipeline steps     │  ResultsViewer                   │
│  + debug log        │                                  │
└─────────────────────┴──────────────────────────────────┘
```

The outer div uses `position: fixed; width: 100%; height: 100%` so it sits on top of the normal page flow and fills the entire viewport. `document.body.style.overflow = 'hidden'` is set while `hasResults` is true to prevent the background from scrolling.

### JSX variable extraction

To avoid duplicating the pipeline steps and debug log markup across both layout branches, they are extracted as JSX variables inside the component body:

```typescript
const pipelineSteps = ( <> {/* steps 1–4 */} </> )
const debugLogEl = ( <DebugLog ... /> )
```

Both layout branches (`hasResults` and not) then reference these variables directly. This keeps the conditional rendering logic clean while ensuring only one copy of each section exists.

---

## Conditional Rendering

- Step 2 (configure) is shown only when video is uploaded and no anchor frames exist yet.
- Step 3 (annotate) + step 4 (pipeline) are shown when `anchorFrames.length > 0`.
- When `hasResults` is true: full-bleed two-column layout (pipeline panel + results panel).
- When `hasResults` is false: standard scrolling `.container` layout with `.main-content` and `.activity-sidebar`.
- Status bar is shown when `status` or `error` is non-empty (pipeline layout only).
