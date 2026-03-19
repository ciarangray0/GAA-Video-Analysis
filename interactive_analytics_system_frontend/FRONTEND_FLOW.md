# GAA Video Analysis Front-End Flow

## Overview
This document explains the full flow of the front-end, detailing component usage, data transfer, API calls, and grouping components by functionality. It also highlights redundant or unused code.

---

## 1. Main Entry & Routing
- **index.tsx** (pages/index.tsx): The main entry point. Orchestrates the workflow and state for the pipeline.
- Uses Next.js for routing.

---

## 2. Component Groups & Pipeline Steps

### A. Video Upload
- **VideoUploader.tsx**
  - Used in index.tsx as `<VideoUploader onUploadSuccess={...} />`
  - Handles file selection and upload.
  - Calls `uploadVideo(file)` from lib/api.ts (POST `/videos`).
  - On success, returns `VideoMetadata` and the file to index.tsx.
  - Data passed: `VideoMetadata`, `File`.

### B. Anchor Frame Configuration & Annotation
- **AnchorFrameAnnotator.tsx**
  - Used in index.tsx as `<AnchorFrameAnnotator ... />`.
  - Lets user annotate anchor frames (points/lines) for pitch mapping.
  - Handles annotation mode, frame navigation, and auto-saving to localStorage.
  - Data passed: `AnchorFrame[]`, `VideoMetadata`, current frame index.
  - On annotation change, calls `onAnchorFramesChange` in index.tsx, marking pipeline steps as stale.

### C. Pipeline Processing
- **PipelineSteps.tsx**
  - Used in index.tsx as `<PipelineSteps ... />`.
  - Manages the multi-step pipeline:
    - **Step A**: Tracking
      - Calls `trackVideo(video_id)` (POST `/videos/{video_id}/track`).
      - Result: `{ frames_processed, tracks, num_detections }`.
    - **Step B**: Homography computation
      - Calls `computeHomographiesV2(video_id, annotations)` (POST `/videos/{video_id}/homographies/v2`).
      - Result: `{ frames, per_frame_count, info }`.
    - **Step C**: Player mapping
      - Calls `mapPlayers(video_id)` (POST `/videos/{video_id}/map_players`).
      - Result: `PlayerPosition[]`.
    - **Step D**: Trajectory interpolation
      - Calls `interpolateTrajectories(video_id, startFrame, endFrame)` (POST `/videos/{video_id}/interpolate`).
      - Result: `{ frames_generated, method }`.
    - **Diagnostics**: Per-frame mapping
      - Calls `GET /videos/{video_id}/diagnostics/per-frame-mapping`.
  - Handles marking steps as stale/running, error/status updates, and logs API calls.
  - Data passed: Results of each step, anchor frames, video metadata.

### D. Results Display
- **ResultsViewer.tsx**
  - Used in index.tsx as `<ResultsViewer ... />`.
  - Displays processed player positions, frame navigation, pitch overlays, and anchor frame quality.
  - Data passed: `PlayerPosition[]`, `VideoMetadata`, `File`, anchor frames, frame indices, FPS.

### E. Debugging & Logging
- **DebugLog.tsx**
  - Used in index.tsx as `<DebugLog ... />`.
  - Shows API call logs, pipeline state, and step results.
  - Data passed: log entries, video metadata, step results.

---

## 3. Data Flow & Transfer

### Upload
- User uploads video in VideoUploader.
- `onUploadSuccess` passes `VideoMetadata` and `File` to index.tsx.

### Anchor Frames
- index.tsx generates anchor frames based on interval and video metadata.
- AnchorFrameAnnotator receives anchor frames and allows annotation.
- On annotation change, index.tsx marks pipeline steps as stale.

### Pipeline Steps
- PipelineSteps receives video metadata, anchor frames, and step results.
- Each step triggers an API call and updates results in index.tsx.
- Results are passed to ResultsViewer and DebugLog.

### Results
- ResultsViewer displays player positions and frame overlays.
- DebugLog shows API call history and pipeline state.

---

## 4. API Calls Used
- `POST /videos` (uploadVideo)
- `POST /videos/{video_id}/track` (trackVideo)
- `POST /videos/{video_id}/homographies/v2` (computeHomographiesV2)
- `POST /videos/{video_id}/map_players` (mapPlayers)
- `POST /videos/{video_id}/interpolate` (interpolateTrajectories)
- `GET /videos/{video_id}/players` (getPlayerPositions)
- `GET /videos/{video_id}/diagnostics/per-frame-mapping`
- `GET /videos/{video_id}/frame/{frame_idx}` (frame image for annotation)
- `GET /videos/{video_id}/frames/{frame_idx}/warped` (warped frame for results)

---

## 5. Component Functionality Groups

- **Upload & Metadata**: VideoUploader
- **Annotation**: AnchorFrameAnnotator
- **Pipeline Processing**: PipelineSteps
- **Results Display**: ResultsViewer
- **Debug/Logging**: DebugLog

---

## 6. Data Transfer Map

- VideoUploader → index.tsx: `VideoMetadata`, `File`
- index.tsx → AnchorFrameAnnotator: `AnchorFrame[]`, `VideoMetadata`, current frame index
- AnchorFrameAnnotator → index.tsx: updated `AnchorFrame[]`
- index.tsx → PipelineSteps: `VideoMetadata`, `AnchorFrame[]`, step results
- PipelineSteps → index.tsx: step results, homography frame indices, player positions
- index.tsx → ResultsViewer: player positions, video metadata, file, anchor frames, frame indices, FPS
- index.tsx → DebugLog: log entries, video metadata, step results

---

## 7. Redundant/Unused Code
- No obvious redundant code in the main pipeline. All components are used in index.tsx and participate in the workflow.
- If any helper functions or legacy API calls exist in lib/api.ts or other files but are not referenced in index.tsx or components, they may be redundant.
- LocalStorage restore logic in index.tsx and AnchorFrameAnnotator is only used if the user uploads the same video file again.

---

## 8. Summary
- The front-end is a modular, step-driven workflow.
- Data flows from upload → annotation → pipeline → results.
- API calls are logged and errors/statuses are surfaced to the user.
- All main components are actively used; helper code should be reviewed for redundancy if not referenced.

