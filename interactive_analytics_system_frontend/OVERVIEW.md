# Frontend Overview

A Next.js single-page application that guides the user through a five-step pipeline: upload a GAA football video, configure and annotate anchor frames, run the backend pipeline, and view the resulting 2D player tracking overlay alongside the original video.

---

## 5-Step User Journey

| Step | UI Element | What happens |
|------|-----------|-------------|
| **1. Upload** | `VideoUploader` | User selects an MP4; `POST /videos` returns metadata (fps, num_frames, duration). |
| **2. Configure** | Inline in `index.tsx` | User picks an anchor frame interval (0.5s–10s). `generateAnchorFrames` creates the frame list. Saved annotations from a previous session are offered for restoration. |
| **3. Annotate** | `AnchorFrameAnnotator` | For each anchor frame: user clicks pitch keypoints and/or line segments. Annotations auto-saved to `localStorage`. |
| **4. Pipeline** | `PipelineSteps` | Steps A–D: tracking → homography → player mapping → interpolation. Each step calls one or more backend API endpoints. |
| **5. Results** | `ResultsViewer` | Side-by-side video + 2D pitch canvas. Playback at adjustable speeds, frame-by-frame stepping, BotSort overlay toggle, team classification (jersey-colour analysis with per-track override), KPI computation, and an analysis trim slider to exclude trailing frames. |

---

## Component Tree

```
index.tsx (Home)
 ├── VideoUploader            (step 1)
 ├── AnchorFrameAnnotator     (step 3, shown when anchorFrames.length > 0)
 ├── PipelineSteps            (step 4)
 ├── ResultsViewer            (step 5, shown when playerPositions.length > 0)
 └── DebugLog                 (sidebar — API call log + pipeline summary)
```

When `playerPositions.length > 0` (i.e. step D has completed), the page switches from the standard scrolling layout to a **full-bleed two-column layout**: a narrow fixed pipeline panel on the left and a full-width scrollable results panel on the right. See `pages/INDEX.md` for layout details.

---

## State Ownership in `index.tsx`

All cross-component state lives in `Home`. Child components receive data as props and report back via callbacks.

| State variable | Type | Purpose |
|----------------|------|---------|
| `videoFile` | `File \| null` | The uploaded video file (used for the HTML video element) |
| `videoMetadata` | `VideoMetadata \| null` | fps, num_frames, width, height, duration_seconds, video_id |
| `anchorInterval` | `number` | Seconds between auto-generated anchor frames |
| `anchorFrames` | `AnchorFrame[]` | All anchor frames with their annotations |
| `currentAnchorIdx` | `number` | Which anchor frame is active in the annotator |
| `stepAResult` | object \| null | Tracking result (frames_processed, tracks, num_detections) |
| `stepBResult` | object \| null | Homography result (anchor frames, per_frame_count, info) |
| `stepCResult` | object \| null | Mapping result (positions, total) |
| `stepDResult` | object \| null | Interpolation result (frames_generated, method) |
| `staleSteps` | `Set<string>` | Steps that need re-running because upstream data changed |
| `runningSteps` | `Set<string>` | Steps currently in-flight (for button disabled state) |
| `playerPositions` | `PlayerPosition[]` | All player positions (sparse + interpolated) |
| `currentFrame` | `number` | Currently displayed frame in ResultsViewer |
| `processedStartFrame` | `number` | Start of the interpolated range |
| `processedEndFrame` | `number` | End of the interpolated range |
| `homographyFrameIndices` | `number[]` | Anchor frame indices (shown as markers in ResultsViewer) |
| `processedFps` | `number` | FPS used for video/frame sync |

---

## Stale-Step Invalidation Logic

When the user changes annotations after running the pipeline, downstream steps become stale. The `staleSteps` set is managed in `index.tsx`:

- `markStale(steps)` — adds step IDs to the set.
- `clearStale(steps)` — removes step IDs when they complete successfully.
- `handleAnchorFramesChange` — if steps B, C, or D have already run, marks them stale when any annotation changes.
- `stepDoneRef` — a `useRef` (not state) that records whether each downstream step has a result. Used in `handleAnchorFramesChange` to avoid marking as stale when nothing has run yet.

`PipelineSteps` also calls `onStepsMarkedStale` when a step completes (e.g. completing A marks B, C, D stale; completing B marks C, D stale).

---

## `generateAnchorFrames()`

Called when the user clicks "Generate Anchor Frames".

1. Iterates from `seconds=0` to `duration_seconds` in steps of `anchorInterval`.
2. Converts each to a frame index: `frameIdx = Math.floor(seconds * fps)`.
3. Creates `AnchorFrame` objects with empty `points` and `lines`.
4. Checks `localStorage` for previously saved annotations under `"gaa_annotations_{videoFile.name}"`.
5. If found, offers a confirm dialog. On accept, merges saved annotations into the new frame list (matching by `frame_idx`).

---

## `logApiCall(entry)`

Appends an API call log line to `debugLog.current` (a `useRef` array) and syncs it to `debugLogEntries` state so `DebugLog` rerenders. Using `useRef` for the array avoids unnecessary state churn — `setDebugLogEntries` is the only state trigger.

---

## localStorage Persistence

Annotations are auto-saved to `localStorage` under `"gaa_annotations_{videoFile.name}"` every time `anchorFrames` changes (via a `useEffect` in `AnchorFrameAnnotator`). On `generateAnchorFrames`, the saved data is offered for restoration. This allows the user to close the browser and resume annotation without losing work.
