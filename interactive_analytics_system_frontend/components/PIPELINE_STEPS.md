# `PipelineSteps` Component

Renders the four sequential pipeline steps (A–D) with run buttons, result displays, and feedback about quality and progress.

---

## Props

| Prop | Description |
|------|-------------|
| `videoMetadata` | Video info — used for `video_id`, `num_frames`, `fps` |
| `anchorFrames` | Current annotations — used to build the homography request body |
| `stepAResult, stepBResult, stepCResult, stepDResult` | Results from parent; null = step not yet run |
| `staleSteps, runningSteps` | Sets of step IDs for UI state |
| `onStepAComplete .. onStepDComplete` | Callbacks that pass results up to parent |
| `onStepsMarkedStale, onStepsClearedStale` | Callbacks to update parent stale set |
| `onRunningStepChange` | `(step, 'add' | 'remove')` — updates parent running set |
| `onError, onStatusChange` | Pass messages to parent status bar |
| `logApiCall` | Appends to the debug log |

---

## Local State

| State | Description |
|-------|-------------|
| `anchorQuality` | Per-anchor reprojection quality data from `GET .../anchor-quality` |
| `anchorQualityError` | Error string if quality fetch failed |
| `stepBVersion` | Integer counter incremented each time step B completes; used as `?v=N` query param on warped-frame thumbnail `<img>` elements to bust browser cache |
| `sgLongWindow, sgMidWindow, maxVelPx` | Step D smoothing parameters exposed as number inputs |

---

## `apiFetch(url, options)` → `Promise<Response>`

Logging wrapper around `fetch`:
1. Calls `logApiCall(`→ ${method} ${url}`)`.
2. Runs `fetch`.
3. On success: `logApiCall(`← ${status} (${elapsed}ms)`)`.
4. On exception: `logApiCall(`✗ ${error}`)`, re-throws.

Used for all step API calls so every request appears in the debug log.

---

## Step A — Tracking (`runStepA`)

1. Calls `POST /videos/{id}/track` via `apiFetch`.
2. On success, calls `getDetections` to count the total detections.
3. Calls `onStepAComplete({frames_processed, tracks, num_detections})`.
4. Marks B, C, D stale (tracking output changed); clears A stale.

**Result display:** video_id, fps, total frames, detection count, unique track count.

---

## Step B — Homographies (`runStepB`)

**Annotation filter:** a frame is usable if it is not skipped AND has at least one point OR at least one line annotation. Line-only frames are valid because `compute_homographies_with_lines_v3` can work with lines even when no keypoints are present for that specific frame (though ≥4 keypoints per frame are needed — this filtering is done server-side).

```typescript
const validAnnotations = anchorFrames
  .filter(af => !af.isSkipped && (af.points.length > 0 || (af.lines || []).length > 0))
  .map(af => ({ frame_idx: af.frame_idx, points: af.points, lines: af.lines || [] }))
```

1. Calls `POST /videos/{id}/homographies/v3` with the annotation list.
2. Builds `StepBResult` from the response.
3. Calls `onStepBComplete(result, data.frames)`.
4. Increments `stepBVersion` to bust the warped-frame thumbnail cache.
5. Marks C, D stale; clears B stale.
6. **Immediately** fetches `GET .../anchor-quality` and stores in `anchorQuality`.

**Result display:**
- Anchor frame count + per_frame_count.
- Failed-frame list (quality == "bad") in red.
- Per-anchor summary table: keypoints, lines, convergence, repr error (mean + max), warnings.
- Anchor quality per-point breakdown table (sorted by error descending) in an expandable `<details>`.
- Warped-frame thumbnails: original | warped | warped-with-players for each anchor frame. The `?v={stepBVersion}` cache buster ensures the browser re-fetches after a new computation.

---

## Step C — Player Mapping (`runStepC`)

1. Calls `mapPlayers(videoId)` from `lib/api.ts`.
2. Calls `onStepCComplete({positions, total})`.
3. Marks D stale; clears C stale.

**Result display:** total positions mapped, expandable table showing first 20 positions (frame, track_id, x_pitch, y_pitch, source).

---

## Step D — Interpolation (`runStepD`)

Uses the three SG/velocity params from local state (user can adjust before running).

1. Calls `interpolateTrajectories(videoId, 0, num_frames-1, {sgLongWindow, sgMidWindow, maxVelPx})`.
2. Fetches all positions via `getPlayerPositions`.
3. Calls `onStepDComplete(result, allPositions, startFrame, endFrame, fps)`.
4. Calls `onStatusChange('Pipeline complete!')`.

**Params UI:** three `<input type="number">` fields for `sg_long_window`, `sg_mid_window`, `max_vel_px` — rendered above the run button so the user can tune before each run.

**Result display:** interpolated frame count + method string.

---

## UI Utility Functions

| Function | Purpose |
|----------|---------|
| `reprErrorLabel(val)` | `"15px ⚠"` style label with icon |
| `reprErrorColor(val)` | CSS colour string (green/amber/red) by threshold |
| `qualityBadge(q)` | `"✅ good"`, `"⚠️ warning"`, `"❌ bad"` |
| `qualityColor(q)` | CSS colour for quality level |
| `verdictBadge(v)` | `"✓"`, `"⚠"`, `"✗"` for per-keypoint verdict |
| `impactColor(impact)` | CSS colour for "helpful"/"marginal"/"harmful" |

These are pure functions used throughout the result tables.

---

## `validAnnotationCount` (useMemo)

Pre-computes the number of annotatable frames to determine whether the step-B button should be enabled:
```typescript
anchorFrames.filter(af => !af.isSkipped && (af.points.length > 0 || (af.lines || []).length > 0)).length
```
The step-B button is disabled when this is 0.
