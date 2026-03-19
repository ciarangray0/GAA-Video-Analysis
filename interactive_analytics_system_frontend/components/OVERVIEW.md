# Components Overview

Five components implement the step-by-step UI. All are pure functional React components with hooks; none maintain global state (all state either lives in `index.tsx` or is local UI state owned by the component itself).

---

## Component Summary

| Component | Role | Key props in | Key callbacks out |
|-----------|------|-------------|-------------------|
| `VideoUploader` | File input + upload to backend | — | `onUploadSuccess(metadata, file)` |
| `AnchorFrameAnnotator` | Frame display + annotation UI | `videoMetadata`, `anchorFrames`, `currentAnchorIdx` | `onAnchorFramesChange(frames)`, `onCurrentIdxChange(idx)` |
| `PipelineSteps` | Steps A–D runner + results display | `videoMetadata`, `anchorFrames`, step results, stale/running sets | `onStepAComplete`, `onStepBComplete`, `onStepCComplete`, `onStepDComplete`, stale/running change callbacks |
| `ResultsViewer` | Side-by-side video + pitch playback | `videoMetadata`, `videoFile`, `playerPositions`, `currentFrame`, frame range info | `onFrameChange(frame)` |
| `DebugLog` | API activity log + pipeline summary | `entries`, step results | `onClear()` |

---

## `VideoUploader`
Renders a file input and upload button. On file selection, calls `uploadVideo(file)` from `lib/api.ts`, then calls `onUploadSuccess(metadata, file)` with the `VideoMetadata` response. Handles errors by displaying them inline.

## `AnchorFrameAnnotator`
The most complex component. Displays the video frame on a `<canvas>` element with annotation overlays, and a pitch diagram canvas for selecting corresponding pitch locations. Supports two annotation modes (point and line). Full documentation: `ANCHOR_FRAME_ANNOTATOR.md`.

## `PipelineSteps`
Renders four collapsible pipeline step panels (A–D) with run buttons, results tables, and debug information. Each step fires an API call and reports the result upward. Manages its own `anchorQuality` and `stepBVersion` state. Full documentation: `PIPELINE_STEPS.md`.

## `ResultsViewer`
Renders the video + 2D pitch canvas side by side. Manages playback via `requestAnimationFrame`, syncs the HTML video element to the current frame, and redraws the pitch canvas via `drawPitch`. Full documentation: `RESULTS_VIEWER.md`.

## `DebugLog`
Scrollable sidebar showing timestamped API log entries (arrows, HTTP status codes, timing). Also shows a summary table of pipeline step results when available.
