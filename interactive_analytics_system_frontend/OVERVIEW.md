# Frontend Overview

This document explains how the frontend is structured and why it was built that way. If you are new to React or Next.js, read from the top — every concept is explained as it comes up.

---

## What is this app?

This is a web app that lets you upload a GAA football video and get back a 2D "bird's eye" map of where every player was during the clip. Think of it like turning a camera recording into a top-down tactical diagram — like the kind you'd see on a coaching whiteboard.

The user walks through five steps in order: upload a video, configure some settings, annotate the pitch, run the backend analysis, then view the results.

---

## What is Next.js?

Next.js is a framework built on top of React. React itself just handles building UIs out of small reusable pieces called **components**. Next.js adds routing (pages), server-side rendering, and build tooling on top.

This app is a **single-page application (SPA)** — meaning it only has one page (`pages/index.tsx`) and everything happens inside that one page without any navigation to different URLs. The five "steps" are just different sections of the UI appearing and disappearing based on the app's current state.

---

## What is a component?

In React, a **component** is just a function that returns HTML-like code (called JSX). Every time the data it depends on changes, React re-runs that function and updates what you see on screen.

```typescript
function VideoUploader({ onUploadSuccess }) {
  // ...returns JSX like <div>, <button>, etc.
}
```

This app has five main components, each responsible for one part of the UI. They are all wired together inside `index.tsx`, which is the "parent" that owns the data and passes it down.

---

## The 5-Step User Journey

| Step | UI Element | What the user does |
|------|-----------|-------------------|
| **1. Upload** | `VideoUploader` | Picks an MP4 file; the backend receives it and sends back info about the video (FPS, duration, etc.) |
| **2. Configure** | Inline in `index.tsx` | Picks how often to sample "anchor frames" — e.g. every 2 seconds — and clicks "Generate Anchor Frames" |
| **3. Annotate** | `AnchorFrameAnnotator` | For each anchor frame, clicks where pitch landmarks are in the video image, and where those same landmarks sit on a 2D pitch diagram. This tells the system how to "unfold" the camera perspective. |
| **4. Pipeline** | `PipelineSteps` | Runs four backend steps: tracking (A), homography computation (B), player mapping (C), and interpolation (D). Each has its own "Run" button. |
| **5. Results** | `ResultsViewer` | Watches the video and a live 2D pitch view side-by-side. Can classify players by team jersey colour, step through frames, and compute spatial KPIs. |

---

## Component Tree

Think of the component tree like a family tree. `index.tsx` (the "Home" component) is the parent. All other components are children — they receive data from Home and report events back up.

```
index.tsx (Home)
 ├── VideoUploader            (step 1)
 ├── AnchorFrameAnnotator     (step 3, appears once anchor frames are generated)
 ├── PipelineSteps            (step 4)
 ├── ResultsViewer            (step 5, appears once player tracking is done)
 └── DebugLog                 (sidebar — shows API call log and pipeline summary)
```

### Support modules

These are not components — they are plain TypeScript files with helper functions that components import and use.

| File | What it does |
|------|-------------|
| `lib/api.ts` | All the HTTP calls to the backend |
| `lib/pitch.ts` | Draws the pitch diagram on a `<canvas>` element |
| `lib/constants.ts` | Numbers like pitch width in metres, canvas size in pixels |
| `utils/canvasUtils.ts` | Draws the crosshair marker when you annotate a point |
| `utils/formatters.ts` | Turns homography quality numbers into coloured labels |
| `utils/kpiUtils.ts` | Aggregates KPI data into readable summaries |
| `types/index.ts` | TypeScript "shape" definitions for every data object |

Once the player data arrives (i.e. step D completes), the page switches from a normal scrolling layout to a **full-bleed two-column layout**: a narrow panel on the left for the pipeline controls, and the full width of the right side for the results view. This gives the results as much screen real estate as possible.

---

## State Ownership in `index.tsx`

### What is "state"?

In React, **state** is data that can change over time and should cause the UI to update when it does. You declare state with `useState`:

```typescript
const [videoMetadata, setVideoMetadata] = useState(null)
```

Here, `videoMetadata` starts as `null`. When the upload completes, the app calls `setVideoMetadata(result)` and React re-renders the component with the new data.

### Why does all the state live in `index.tsx`?

Because multiple components need access to the same data. For example, `AnchorFrameAnnotator` needs the video metadata, and so does `PipelineSteps`. Rather than each component fetching the same data separately, the parent (`index.tsx`) fetches it once and passes it down to both. This pattern is called **"lifting state up"** — you move the state to the closest ancestor that needs to share it.

### All the state variables

| Variable | Type | What it holds |
|---|---|---|
| `videoFile` | `File \| null` | The raw video file the user picked (used by the HTML video element for playback) |
| `videoMetadata` | `VideoMetadata \| null` | Info from the backend: fps, total frames, width, height, duration, and the video's ID |
| `anchorInterval` | `number` | How many seconds apart to space the anchor frames |
| `anchorFrames` | `AnchorFrame[]` | The list of anchor frames, each with its annotation points and lines |
| `currentAnchorIdx` | `number` | Which anchor frame the annotator is currently showing |
| `stepAResult` | object \| null | What step A (tracking) returned |
| `stepBResult` | object \| null | What step B (homography) returned |
| `stepCResult` | object \| null | What step C (player mapping) returned |
| `stepDResult` | object \| null | What step D (interpolation) returned |
| `staleSteps` | `Set<string>` | Steps that are "out of date" and need to be re-run |
| `runningSteps` | `Set<string>` | Steps currently waiting on the backend (used to disable buttons) |
| `playerPositions` | `PlayerPosition[]` | All player positions on the 2D pitch (sparse + interpolated) |
| `currentFrame` | `number` | The frame number currently shown in ResultsViewer |
| `processedStartFrame` | `number` | First frame of the interpolated range |
| `processedEndFrame` | `number` | Last frame of the interpolated range |
| `homographyFrameIndices` | `number[]` | Frame indices of anchor frames (shown as markers in ResultsViewer) |
| `processedFps` | `number` | FPS used to sync video and frame counter |

---

## Stale-Step Invalidation Logic

### The problem this solves

Imagine the user runs the full pipeline (steps A through D), then goes back and changes their annotations. The homography, player mapping, and interpolation results are now wrong — they were computed from the old annotations. The app needs to warn the user that those steps need re-running.

### How it works

There is a `staleSteps` state variable, which is a `Set` (like an array, but with no duplicates) of step IDs like `"B"`, `"C"`, `"D"`.

Two helper functions manage it:

- `markStale(steps)` — adds step IDs to the set. For example: `markStale(["B", "C", "D"])`.
- `clearStale(steps)` — removes step IDs when they complete successfully.

When the user changes any annotation, the function `handleAnchorFramesChange` checks whether steps B, C, or D have already produced results. If they have, it calls `markStale(["B", "C", "D"])`. The UI then shows those steps with a "stale" warning.

There is also a `stepDoneRef` — a `useRef` (explained below) that tracks whether each step has ever completed. It is used here instead of state to avoid triggering unnecessary re-renders.

`PipelineSteps` also triggers `onStepsMarkedStale` after completing a step, so completing step A automatically marks B, C, and D as stale (since they need to be re-run with the new tracking data).

### What is `useRef`?

`useRef` is like `useState` but without the re-render. Use it when you need to remember a value across renders but changing it shouldn't cause the UI to update. Here, `stepDoneRef` stores whether each step has run — we just need that fact available when checking staleness, not to display it anywhere.

---

## `generateAnchorFrames()`

This function runs when the user clicks "Generate Anchor Frames". It creates the list of frames that will need to be annotated.

Here is what it does, step by step:

1. Starts at `seconds = 0` and counts up to the video duration in steps of `anchorInterval`.
2. Converts each second value to a frame index: `frameIdx = Math.floor(seconds * fps)`.
   - For example: 4.0 seconds at 25 fps = frame 100.
3. Creates an `AnchorFrame` object for each, with empty `points` and `lines` arrays (no annotations yet).
4. Checks `localStorage` (the browser's built-in storage that survives page refreshes) for annotations previously saved under the key `"gaa_annotations_{videoFile.name}"`.
5. If saved annotations are found, it pops up a confirm dialog asking whether to restore them. If the user says yes, it merges the saved annotations into the new frame list by matching `frame_idx` values.

This means the user can close the browser mid-annotation and pick up exactly where they left off.

---

## `logApiCall(entry)`

Every time a backend API call is made, this function records it in a log so the user can see what happened (status codes, timing, errors). It keeps that log in `debugLog.current` — a `useRef` array — and then copies it into `debugLogEntries` state so the `DebugLog` component re-renders and shows the new entry.

Why use a `useRef` for the array? Because appending to the array should not itself trigger a re-render — only the `setDebugLogEntries` call at the end should. Using `useRef` for the underlying storage avoids React treating every intermediate step as a state change.

---

## localStorage Persistence

Annotations are automatically saved to `localStorage` every time `anchorFrames` changes. This is done via a `useEffect` in `AnchorFrameAnnotator`:

```typescript
useEffect(() => {
  localStorage.setItem("gaa_annotations_" + fileName, JSON.stringify(anchorFrames))
}, [anchorFrames])
```

Think of `useEffect` as "run this code whenever these values change". Any time the annotations array changes, the effect fires and writes the latest data to browser storage.

This means the user never needs to manually save — annotations are persisted automatically and can be restored the next time they open the app with the same video file.
