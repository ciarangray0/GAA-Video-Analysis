# `pages/index.tsx` — The Root Page

This is the top-level file of the application. In Next.js, every file in the `pages/` folder becomes a URL route. `pages/index.tsx` is the root route (`/`) — the page the user sees when they open the app. It acts as the "brain" that connects all the other components together and owns all the shared data.

---

## What does "root page" mean?

Think of the application as a tree:

```
pages/index.tsx         ← you are here (owns all the data)
├── VideoUploader       ← handles file selection + upload
├── AnchorFrameAnnotator ← annotation interface
├── PipelineSteps       ← run A/B/C/D buttons
└── ResultsViewer       ← video + pitch playback
```

Data flows DOWN from the root page to each component as props. When a component wants to change something, it calls a callback function that the root page gave it — the root page then updates its own state, which flows back down. This is called "one-way data flow" and it means you can always look at `index.tsx` to understand the full state of the application.

---

## useState — tracking values that change

```typescript
const [videoFile, setVideoFile] = useState<File | null>(null)
const [anchorFrames, setAnchorFrames] = useState<AnchorFrame[]>([])
```

`useState` creates a variable (`videoFile`) and a setter function (`setVideoFile`). When you call the setter, React re-renders the component with the new value. You never assign directly (not `videoFile = someFile`) — only the setter triggers a re-render.

Here is every piece of state owned by the root page:

| Variable | Starts as | What it is |
|----------|-----------|-----------|
| `videoFile` | `null` | The raw video file the user selected. Passed to `ResultsViewer` so it can play the video locally. |
| `videoMetadata` | `null` | Info returned by the server after upload: `video_id`, fps, frame count. `video_id` is used in every subsequent API call. |
| `anchorInterval` | `1` | How many seconds apart to place anchor frames (user can change this before generating). |
| `anchorFrames` | `[]` | The complete list of annotation frames. Each has `frame_idx`, `isSkipped`, `points` (keypoint pairs), `lines` (line annotations). This is the core data the pipeline runs on. |
| `currentAnchorIdx` | `0` | Which annotation frame the annotator is currently showing. |
| `stepAResult` | `null` | Results from the tracking step, or `null` if not run yet. |
| `stepBResult` | `null` | Results from the homography step. |
| `stepCResult` | `null` | Results from the player mapping step. |
| `stepDResult` | `null` | Results from the interpolation step. |
| `staleSteps` | `new Set()` | A set of step letters (`"A"`, `"B"`, `"C"`, `"D"`) that are out of date. Shown as "STALE" badges in the pipeline panel. |
| `runningSteps` | `new Set()` | Steps currently executing. Used to disable their buttons. |
| `playerPositions` | `[]` | The full list of player positions after step D. Passed to `ResultsViewer`. |
| `currentFrame` | `0` | Which video frame is currently displayed in `ResultsViewer`. |
| `processedStartFrame`, `processedEndFrame` | `0` | The range of frames the interpolation step covered. |
| `homographyFrameIndices` | `[]` | Which frame indices are anchor frames (returned from step B). |
| `processedFps` | `25` | The fps value used during processing. |
| `status`, `error` | `''` | User-facing messages shown in the status bar. |
| `debugLogEntries` | `[]` | A copy of the API call log, used to display the debug log panel. |

---

## useRef — values that don't cause re-renders

```typescript
const debugLog = useRef<string[]>([])
const stepDoneRef = useRef({ B: false, C: false, D: false })
```

`useRef` is like a box you can store a value in, but changing the value does NOT cause a re-render. Two important uses here:

**`debugLog`:** Every API call gets logged (e.g. `"→ POST /track"`, `"← 200 (1234ms)"`). We use a ref for the raw array and a separate state (`debugLogEntries`) for the rendered copy. Why? If we stored the log only in state, every `console.log` equivalent would trigger a full re-render. The ref holds the accumulating array; state only gets updated when we want the UI to reflect it.

**`stepDoneRef`:** Tracks whether each downstream step has ever produced results. Used to avoid a subtle bug: when the user first adds annotations, we should NOT immediately mark steps stale (they have never run — there is nothing to go stale). `stepDoneRef` is checked in `handleAnchorFramesChange` before marking anything stale. Because it is a ref, updating it does not cause a re-render, avoiding an infinite loop.

---

## `generateAnchorFrames()` — creating the annotation list

This function runs when the user clicks "Generate Anchor Frames". It creates evenly-spaced frames for the user to annotate:

```typescript
for (let seconds = 0; seconds <= duration_seconds; seconds += anchorInterval) {
  const frameIdx = Math.floor(seconds * fps)
  frames.push({ frame_idx: frameIdx, isSkipped: false, points: [], lines: [] })
}
```

For a 30-second video at 25fps with a 1-second interval, this creates frames at indices 0, 25, 50, 75, ... 750. Each starts with empty `points` and `lines` — the user fills these in via `AnchorFrameAnnotator`.

**localStorage restore:** After generating frames, the function checks `localStorage` for previously saved annotations under the key `"gaa_annotations_{videoFilename}"`. If found, it shows a confirm dialog. On confirmation, it merges the saved data: for each generated frame, if a saved frame with the same `frame_idx` exists, it restores `isSkipped`, `points`, and `lines` from the save. This means annotations survive page refreshes.

---

## `handleAnchorFramesChange(frames)` — reacting to annotation changes

`AnchorFrameAnnotator` calls this function whenever the user adds, removes, or modifies an annotation. The root page:

1. Updates `anchorFrames` state with the new array.
2. Checks `stepDoneRef` — if steps B, C, or D have ever run, marks them stale (new annotations → old homographies are outdated).

The `stepDoneRef` check prevents false stale-marking on first use: when the page loads and you add your very first annotation, steps B/C/D have never run, so there is nothing to go stale.

---

## `markStale(steps)` and `clearStale(steps)` — updating a Set safely

```typescript
const markStale = (steps: string[]) => {
  setStaleSteps(prev => {
    const next = new Set(prev)
    steps.forEach(s => next.add(s))
    return next
  })
}
```

`staleSteps` is a `Set` (like an array but with no duplicates and fast "does it contain X?" checks). React state must be replaced rather than mutated — you cannot do `staleSteps.add("B")` directly because React would not notice.

The **functional updater form** (`prev => ...`) is used here. Instead of reading `staleSteps` directly from the component's closure (which might be stale if multiple state updates happen in quick succession), we receive the LATEST version as `prev`. This prevents a subtle bug where two rapid updates could overwrite each other.

Step by step:
1. `prev` = the current set.
2. `const next = new Set(prev)` = make a copy (we cannot mutate `prev`).
3. Add or remove from `next`.
4. Return `next` — React sees a new Set and re-renders.

---

## `logApiCall(entry)` — adding to the debug log

```typescript
const logApiCall = (entry: string) => {
  debugLog.current = [...debugLog.current, entry]
  setDebugLogEntries([...debugLog.current])
}
```

`debugLog.current` is the ref that holds all log entries. We spread it into a new array (`[...debugLog.current, entry]`) rather than pushing to it. Why? Because `.push()` mutates the existing array in place. Using spread creates a brand-new array, which ensures React sees a new reference when we call `setDebugLogEntries`. React compares by reference — if it is the same array object, React might skip the re-render.

---

## Step D completion — setting the initial frame

```typescript
onStepDComplete={(result, positions, start, end, fps) => {
  setPlayerPositions(positions)
  // ... other state updates ...
  const firstFrame = positions.length > 0
    ? Math.min(...positions.map(p => p.frame_idx))
    : start
  setCurrentFrame(firstFrame)
}}
```

After the pipeline finishes, `ResultsViewer` needs to start at a useful frame. Frame 0 might be outside the processed range (the pipeline might only have positions starting at frame 50). So instead of starting at 0, we find the lowest `frame_idx` that actually has player data using `Math.min(...positions.map(p => p.frame_idx))`. This ensures the pitch canvas shows dots immediately rather than appearing empty.

`Math.min(...array)` is JavaScript's way of finding the minimum value in an array. The `...` (spread operator) unpacks the array into individual arguments.

---

## `hasResults` and the full-screen layout

```typescript
const hasResults = playerPositions.length > 0 && videoMetadata !== null && videoFile !== null
```

`hasResults` is true once the full pipeline has run and there are player positions to display. When this is true, the page completely changes its layout from a scrolling pipeline view to a full-screen two-column view:

```
┌─────────────────────┬──────────────────────────────────┐
│  pipeline panel     │  results panel                   │
│  (380px, fixed)     │  (fills remaining space)         │
│  steps + debug log  │  ResultsViewer                   │
└─────────────────────┴──────────────────────────────────┘
```

The outer `div` uses `position: fixed; width: 100%; height: 100%` to sit on top of the normal page flow and fill the entire viewport. `document.body.style.overflow = 'hidden'` prevents the background from scrolling while this layout is active.

### JSX variable extraction — avoiding copy-paste

Both layout branches (with results and without) need the pipeline steps panel and the debug log. Rather than duplicating that JSX in two places (and risking them getting out of sync), they are stored as variables:

```typescript
const pipelineSteps = (
  <>
    {/* All 4 step panels */}
  </>
)
const debugLogEl = (<DebugLog entries={debugLogEntries} />)
```

Both layout branches then reference `{pipelineSteps}` and `{debugLogEl}`. This is a common pattern for keeping conditional rendering clean.

---

## Conditional rendering — what shows when

React renders different UI based on the current state. Here is the progression:

1. **Nothing uploaded yet:** Show `VideoUploader` only.
2. **Video uploaded, no anchor frames:** Show the "Configure anchor interval" section.
3. **Anchor frames generated:** Show `AnchorFrameAnnotator` + `PipelineSteps`.
4. **Pipeline complete (`hasResults`):** Switch to full-screen two-column layout with `ResultsViewer` on the right.

Each stage shows only what is relevant, guiding the user through the workflow in order.

The status bar (showing `status` or `error` messages) is only shown in the pipeline layout (stages 1–3) — in the full-screen results layout, the pipeline panel handles its own feedback.
