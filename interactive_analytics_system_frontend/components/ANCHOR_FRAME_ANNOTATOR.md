# `AnchorFrameAnnotator` Component

This is the annotation interface — the screen where a user teaches the system which points on the video frame correspond to which points on the pitch diagram. Think of it as a "matching game": you click a spot on the video, then click the matching spot on the pitch map. The system collects enough of these pairs to calculate a homography (a mathematical transformation that maps video pixels to real-world pitch coordinates).

---

## What is a component?

In React, a component is just a function that returns HTML-like code (called JSX). Every time its data changes, React re-runs the function and updates what you see on screen. `AnchorFrameAnnotator` is a component — it renders two canvases side by side (one for the video frame, one for the pitch diagram) and handles all the user interaction needed to create annotations.

---

## Props — data passed in from the parent

Think of props like arguments to a function. The parent component calls `<AnchorFrameAnnotator videoMetadata={...} anchorFrames={...} ... />` and passes in this data:

| Prop | What it is |
|------|-----------|
| `videoMetadata` | Info about the uploaded video — its ID (so we can fetch frames from the server), fps, and total frame count |
| `videoFilename` | The filename, used as a key for auto-saving annotations in the browser |
| `anchorFrames` | The full list of annotations so far. This component reads it but does NOT modify it directly — it tells the parent when changes happen |
| `currentAnchorIdx` | Which frame in the list we're currently looking at (0 = first, 1 = second, etc.) |
| `onAnchorFramesChange` | A function the parent gives us. We call it whenever annotations change. This is how a child component "talks back" to its parent in React |
| `onCurrentIdxChange` | A function to tell the parent "the user navigated to a different frame" |

**Why does the parent own the data?** Because multiple components (the annotator AND the pipeline steps panel) both need to read `anchorFrames`. Putting the data in the parent means both components always see the same thing. This is called "lifting state up".

---

## useState — how React tracks changing values

```typescript
const [loadingFrame, setLoadingFrame] = useState(false)
const [annotationMode, setAnnotationMode] = useState<'point' | 'line'>('point')
const [zoom, setZoom] = useState(1)
```

Think of `useState` like a variable with a built-in alarm. `loadingFrame` is the current value (starts as `false`). `setLoadingFrame` is the function you call to change it. When you call `setLoadingFrame(true)`, React knows something changed and re-renders the component with the new value. You must never write `loadingFrame = true` directly — React wouldn't notice and the screen wouldn't update.

Here is every piece of local state this component tracks:

| State | Starts as | What it tracks |
|-------|-----------|----------------|
| `loadingFrame` | `false` | True while the frame image is downloading from the server |
| `annotationMode` | `'point'` | Whether the user is placing keypoints (`'point'`) or drawing pitch lines (`'line'`) |
| `selectedLineId` | `'20m_top'` | Which pitch line is currently selected in line mode |
| `pendingLinePoint1` | `null` | In line mode: the image coordinates of the user's first click (waiting for a second click) |
| `pendingFrameClick` | `null` | In point mode: the image coordinates of the frame click (waiting for the user to click the matching spot on the pitch diagram) |
| `copyStatus` | `''` | A temporary "Copied from previous frame" message shown briefly after a copy action |
| `zoom` | `1` | How zoomed in the frame canvas is (1×, 1.5×, 2×, 3×, or 4×) |
| `canvasDims` | `{w:0, h:0}` | The actual pixel size of the canvas buffer — set when an image loads |
| `hoverPos` | `null` | The pixel coordinates where the mouse is currently hovering (for a live readout display) |

---

## useRef — values that don't trigger re-renders

```typescript
const frameCanvasRef = useRef<HTMLCanvasElement>(null)
const loadingFrameIdxRef = useRef<number>(-1)
```

A `useRef` is like a box you can store a value in, but changing the value does NOT cause a re-render. This is useful for two things:

1. **Accessing DOM elements directly** — `frameCanvasRef.current` gives us the actual `<canvas>` HTML element so we can draw on it using the Canvas 2D API.
2. **Tracking values across renders without triggering them** — `loadingFrameIdxRef` stores which frame is currently loading, but we don't want changing it to cause a re-render.

| Ref | What it points to |
|-----|-------------------|
| `frameCanvasRef` | The `<canvas>` element that displays the video frame |
| `frameImageRef` | The currently loaded image object |
| `pitchDiagramRef` | The `<canvas>` element showing the pitch diagram |
| `importAnnotationsRef` | A hidden `<input type="file">` element (triggered programmatically for JSON import) |
| `loadingFrameIdxRef` | Tracks which frame index the latest fetch is for (see stale-load prevention below) |
| `hasLoadedRef` | Prevents loading the initial frame twice on mount |

---

## Loading a frame: `loadFrameImage(frameIdx)`

When the user navigates to a frame, we need to fetch that frame's image from the server (`GET /videos/{video_id}/frame/{frameIdx}`). Here is how it works and why some of the code might look unusual:

```typescript
loadingFrameIdxRef.current = frameIdx   // remember which frame we're loading
const img = new Image()
img.src = `/videos/${videoId}/frame/${frameIdx}?t=${Date.now()}`
img.onload = () => {
  if (loadingFrameIdxRef.current !== frameIdx) return  // discard stale load
  frameImageRef.current = img
  drawFrameWithPoints()
}
```

**Why the stale-load check?** Imagine the user clicks "next frame" twice very quickly. Two fetches start (for frame 5, then frame 6). Frame 5's network request might finish AFTER frame 6's — this is called a "race condition". Without the check, frame 5 would overwrite frame 6 on the canvas, showing the wrong image. The `loadingFrameIdxRef` stores the most recent requested frame index. When frame 5's `onload` fires, it checks: "is `loadingFrameIdxRef` still 5?" — it's now 6, so it discards the result.

**Why `?t=${Date.now()}`?** Browsers cache images. If you load frame 5, annotate it, then re-load it without this trick, the browser would show the old cached image and not re-fetch. Adding a unique timestamp forces the browser to make a fresh request every time.

---

## Drawing the canvas: `drawFrameWithPoints()`

This function erases and redraws the entire frame canvas from scratch every time annotations change. This is the standard approach with the HTML Canvas API — you can't "edit" a canvas like you edit the DOM; you redraw it.

**Buffer sizing:**
```typescript
const scale = Math.min(1, 1600 / img.naturalWidth)
const newW  = Math.round(img.naturalWidth  * scale)
const newH  = Math.round(img.naturalHeight * scale)
```
The canvas buffer is sized to at most 1600px wide (keeping the image's proportions). Earlier versions used 1000px, but that was too coarse for precise goal-area annotations. 1600px is a balance between precision and memory usage.

**Drawing order** (later items draw on top of earlier ones):
1. Draw the scaled video frame image onto the canvas.
2. For each line annotation: draw a dashed coloured line between its two endpoints, plus crosshair markers at both ends. Cyan = horizontal pitch lines, orange = vertical.
3. If a first line point is pending: draw a yellow crosshair labelled "←2nd point" to show where the first click landed.
4. For each keypoint: draw a green crosshair with the pitch vertex label.

**Coordinate scaling:** annotation coordinates are stored in original image pixels (e.g. "1920×1080 video → click at pixel 960, 540"). The canvas buffer is smaller (max 1600px wide). To draw a stored annotation in the right place on the canvas:
```typescript
const imgScaleX = canvas.width  / img.naturalWidth
const imgScaleY = canvas.height / img.naturalHeight
```
Separate X and Y scales handle images whose aspect ratio creates non-round numbers.

---

## Converting a mouse click to image coordinates

When the user clicks the canvas, we need to know which original image pixel they clicked. The canvas might be displayed at a different size than its internal resolution (because of zoom or screen scaling). This conversion handles that:

```typescript
const rect = canvas.getBoundingClientRect()   // the canvas's position and size on screen
const x = (e.clientX - rect.left) * img.naturalWidth  / rect.width
const y = (e.clientY - rect.top)  * img.naturalHeight / rect.height
```

Step by step:
- `e.clientX - rect.left` = how many screen pixels from the left edge of the canvas the user clicked.
- `/ rect.width` = convert to a fraction (0.0 to 1.0) of the canvas's display width.
- `* img.naturalWidth` = scale that fraction to the original image's pixel width.

The result is the original image pixel coordinate, regardless of how zoomed-in the canvas is.

**The outline trick:** The canvas uses `outline: 2px solid ...` in its CSS instead of `border: 2px solid ...`. This matters because `getBoundingClientRect()` returns the "border-box" — the position including any border thickness. A 2px border makes `rect.left` 2px further right than the actual image content, introducing a small but systematic error in every click coordinate. CSS `outline` is drawn outside the layout box and does NOT affect `getBoundingClientRect`, so there is no offset.

---

## Zoom

```typescript
// Applied as inline CSS on the canvas element:
canvas.style.width  = `${canvas.width  * zoom}px`
canvas.style.height = `${canvas.height * zoom}px`
```

The canvas's internal pixel buffer stays the same size. We just change how large it appears on screen using CSS. The coordinate formula above handles this automatically — `rect.width` already equals `canvas.width * zoom`, so the division cancels it out.

At zoom 2× or higher, `image-rendering: pixelated` is applied in CSS. Without this, the browser would blur the image when scaling it up. `pixelated` keeps individual pixels sharp, which makes precise annotation easier.

---

## How annotations are created: the two-click flow

### Point mode (matching video pixels to pitch coordinates)

This is a two-step process:

**Step 1 — click the video frame:**
The user clicks somewhere on the frame canvas (say, a corner post). `handleFrameClick` converts the click to image coordinates and saves them in `pendingFrameClick`. A yellow crosshair appears on the canvas to show "I'm waiting for the matching pitch point".

**Step 2 — click the pitch diagram:**
The user clicks the matching location on the pitch diagram. `handlePitchDiagramClick` does two things:
- Looks for the nearest named vertex (a corner, goal post, etc.) within 20 screen pixels.
- If no vertex is close enough, finds the nearest point on any `PITCH_LINE_SEGMENTS` within 15 screen pixels, and computes a position along that line segment.

Once a pitch location is found, the annotation is added: `{ x_img, y_img }` (image pixel from step 1) paired with a pitch ID string (from step 2). Then `pendingFrameClick` is cleared.

**Cancelling:** Right-click or pressing Escape clears `pendingFrameClick` and removes the yellow crosshair.

**Why snap to vertices?** Clicking exactly on a specific pixel is hard. Snapping to known pitch features (goal posts, sideline intersections) gives the system precise, semantically meaningful correspondences — "this pixel IS the 45m line intersection" — which makes the homography computation much more accurate.

### Line mode (annotating straight pitch lines)

Instead of point pairs, the user can annotate an entire line (e.g. "the 20m line").

**Step 1:** Select a line from the dropdown (e.g. "20m top"). Then click where that line starts in the video frame.

**Step 2:** Click where that line ends in the video frame. Both clicks are in image pixels. The system stores `(u1, v1, u2, v2)` — the two endpoints in image space. The backend uses line annotations differently from point annotations: it fits the homography so that the line through `(u1,v1)→(u2,v2)` in the image maps to the known pitch line in the real world.

---

## Auto-save to localStorage

Every time `anchorFrames` changes, annotations are saved to the browser's `localStorage` under the key `"gaa_annotations_{videoFilename}"`. When the user generates anchor frames, the system checks `localStorage` for existing data and offers to restore it. This means annotations survive page refreshes.

---

## Copy / Skip / Swap actions

| Action | What it does |
|--------|-------------|
| **Skip frame** | Marks this frame as `isSkipped: true`. Skipped frames are not sent to the homography computation. Useful for frames where the camera is mid-pan or the view is obscured. |
| **Swap frame** | Prompts for a new frame number and replaces the current anchor frame. Clears all existing annotations for that slot. |
| **Copy previous / Copy next** | Copies the `points` and `lines` from the nearest non-skipped neighbour frame that has annotations. Useful when many consecutive frames have similar views. Shows a brief "Copied from..." message (stored in `copyStatus` state). |

---

## Coverage metrics

The annotation interface shows a coverage percentage and dot colour to warn you if your annotations are too clustered in one area of the pitch. Here is the maths:

```
bboxW = max(x) - min(x)  of annotated points (in pitch metres)
bboxH = max(y) - min(y)  of annotated points (in pitch metres)
coveragePercent = bboxW * bboxH / (85 × 140) * 100
```

If this is below 10%, a warning appears. Why? Imagine you only annotated goal-post corners at one end. The system sees correspondences covering only ~10×5m out of an 85×140m pitch. A homography computed from such a tight cluster can look great in the annotated area but be wildly wrong elsewhere — a small rotation error at short range becomes a huge position error when extrapolated across the full pitch.

**Point count colour:** red = fewer than 4 points (not enough for a valid homography), orange = 4–6 points (marginal), green = 7 or more (good).

---

## Import / Export

- **Export:** Converts `anchorFrames` to JSON and triggers a browser file download. Lets you back up annotations or share them.
- **Import:** Reads a JSON file, checks it contains a valid `anchorFrames` array, and merges it into the current list by `frame_idx`. If the frame indices in the file don't match the current anchor frames, a confirmation dialog warns you before overwriting.

---

## Crosshair marker design: `drawCrosshair(ctx, cx, cy, color, label?)`

Each annotation is drawn as a precision crosshair rather than a filled circle. Here is how it is drawn:

1. Dark grey shadow lines (slightly offset) for contrast on any background colour.
2. Coloured arm lines extending 7 pixels outward from a 2-pixel gap around the centre.
3. A 2-pixel filled circle exactly at the click point.
4. An optional text label in a small monospace font with a semi-transparent background badge.

**Why not a filled circle?** Earlier versions used a 5px filled circle. The problem: the circle's area extends 5 pixels in ALL directions from the centre. When you look at it, you perceive the centre of the filled area — not the centre pixel. This makes it look like the annotation is placed slightly off from where you clicked, which was confusing. The crosshair design puts the visual centre precisely on the clicked pixel.
