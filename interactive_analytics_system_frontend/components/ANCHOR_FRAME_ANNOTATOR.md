# `AnchorFrameAnnotator` Component

The annotation interface. Displays a video frame on one canvas and a pitch diagram on an adjacent canvas. The user clicks to create point correspondences (keypoints) or annotate pitch lines. Annotations are auto-saved to `localStorage`.

---

## Props

| Prop | Type | Description |
|------|------|-------------|
| `videoMetadata` | `VideoMetadata` | Contains `video_id` for frame fetch and `fps`/`num_frames` for validation |
| `videoFilename` | `string \| undefined` | Used as the `localStorage` key suffix |
| `anchorFrames` | `AnchorFrame[]` | Complete annotation state — read-only; changes reported via callback |
| `currentAnchorIdx` | `number` | Which frame is currently displayed |
| `onAnchorFramesChange` | function | Called whenever annotations are modified |
| `onCurrentIdxChange` | function | Called when navigation changes the active anchor |

---

## State Variables

| State | Initial | Description |
|-------|---------|-------------|
| `loadingFrame` | `false` | True while the frame image is fetching |
| `annotationMode` | `'point'` | `'point'` or `'line'` |
| `selectedLineId` | `'20m_top'` | Which line ID is active in line mode |
| `pendingLinePoint1` | `null` | Image coords of the first click in a two-click line annotation |
| `pendingFrameClick` | `null` | Image coords of a frame click awaiting a pitch diagram click |
| `copyStatus` | `''` | Transient "Copied from previous" message |
| `zoom` | `1` | Display scale factor (1, 1.5, 2, 3, or 4×) |
| `canvasDims` | `{w:0, h:0}` | Buffer canvas dimensions (set when image loads) |
| `hoverPos` | `null` | Image-space pixel coordinates of the current mouse hover (live readout) |

---

## Refs

| Ref | Description |
|-----|-------------|
| `frameCanvasRef` | The frame buffer canvas element |
| `frameImageRef` | The currently loaded `HTMLImageElement` |
| `pitchDiagramRef` | The pitch diagram canvas element |
| `importAnnotationsRef` | Hidden file input for JSON import |
| `loadingFrameIdxRef` | Tracks which `frame_idx` the most recent `loadFrameImage` call is targeting |
| `hasLoadedRef` | Prevents double-loading the initial frame |

---

## `loadFrameImage(frameIdx)`

Fetches the frame image from `GET /videos/{video_id}/frame/{frameIdx}`.

**Stale-load prevention:** Before starting the fetch, sets `loadingFrameIdxRef.current = frameIdx`. In the `img.onload` callback, checks if `loadingFrameIdxRef.current === frameIdx` before updating `frameImageRef`. If the user navigates to a different frame before the slow load finishes, the earlier load's `onload` fires but the check fails — the stale image is discarded and the canvas is not updated.

The image URL includes `?t=${Date.now()}` to bypass browser caching (ensures the freshest frame is always loaded).

---

## `drawFrameWithPoints()`

Redraws the frame canvas.

**Buffer canvas setup:**
```typescript
const scale = Math.min(1, 1600 / img.naturalWidth)
const newW  = Math.round(img.naturalWidth  * scale)
const newH  = Math.round(img.naturalHeight * scale)
```
The buffer is sized to `min(naturalWidth, 1600) × proportional_height`. The cap at 1600px balances annotation precision against memory (earlier versions used 1000px, which was insufficiently precise for goal-area features).

**Drawing order:**
1. Draw the image scaled to the buffer canvas.
2. For each line annotation: draw a dashed line between the two endpoints (cyan for horizontal, orange for vertical) + crosshair markers at both endpoints.
3. If `pendingLinePoint1` exists: draw a yellow crosshair with "←2nd point" label.
4. For each keypoint: draw a green crosshair with the `pitch_id` label.

Line and keypoint coordinates in the annotation are in **original image pixels**. To position them on the canvas, scale by:
```typescript
const imgScaleX = canvas.width / img.naturalWidth
const imgScaleY = canvas.height / img.naturalHeight
```
Separate X and Y scales handle images whose aspect ratio doesn't divide evenly.

---

## `canvasEventToImageCoords(e)`

Converts a mouse event on the frame canvas to original image pixel coordinates.

```typescript
const rect = canvas.getBoundingClientRect()
const x = (e.clientX - rect.left) * img.naturalWidth  / rect.width
const y = (e.clientY - rect.top)  * img.naturalHeight / rect.height
return { x: Math.round(x), y: Math.round(y) }
```

**The `outline` fix:** The canvas uses CSS `outline: 2px solid ...` rather than `border: 2px solid ...`. `getBoundingClientRect()` returns the **border-box** (including border thickness), so a 2px border would make `rect.left` 2px further right than the content edge, introducing a ~4–8px systematic offset in image space. `outline` is drawn *outside* the layout box and does not affect `getBoundingClientRect`. This fix ensures click coordinates are correct.

**Zoom:** CSS `style.width = canvas.width * zoom` is applied, so `rect.width = canvas.width * zoom`. The formula `img.naturalWidth / rect.width` automatically accounts for zoom without any additional factor.

---

## Click Handling Flow

### Point Mode
1. `handleFrameClick` → calls `canvasEventToImageCoords` → sets `pendingFrameClick`.
2. User clicks pitch diagram → `handlePitchDiagramClick`.
3. Find closest vertex within 20px, OR closest point on any `PITCH_LINE_SEGMENTS` within 15px.
4. If vertex found: `addKeypoint(vertexId)`.
5. If line segment found: compute parametric position along segment → `addKeypoint("line_{name}_x{x}_y{y}")`.
6. `addKeypoint` creates a `PitchPoint`, updates `anchorFrames`, clears `pendingFrameClick`.

### Line Mode
1. `handleFrameClick` on first click → sets `pendingLinePoint1`.
2. `handleFrameClick` on second click → creates `LineAnnotation`, adds to `anchorFrames[currentAnchorIdx].lines` (replacing any existing annotation for the same `line_id`), clears `pendingLinePoint1`.

---

## Copy / Swap / Skip Actions

| Action | Description |
|--------|-------------|
| **Skip Frame** | Toggles `isSkipped`. Skipped frames are excluded from the homography computation. |
| **Swap Frame** | Prompts for a new frame number, replaces the anchor at the current index. Clears all annotations for that slot. |
| **Copy Previous / Copy Next** | Copies `points` and `lines` from the nearest non-skipped neighbour with annotations. Shows a timed status message. Useful for annotating many similar frames. |

---

## Coverage Metrics

Displayed as a coloured dot + percentage in the anchor info bar.

**Coverage % calculation:**
```
bboxW = max(x) - min(x) of annotated points (in pitch meters)
bboxH = max(y) - min(y) of annotated points (in pitch meters)
coveragePercent = bboxW * bboxH / (85 * 140) * 100
```
If `coveragePercent < 10%`, the annotations are considered "clustered" and a warning is shown. Clustered annotations lead to poorly conditioned homographies because the system can't distinguish rotation from translation.

**Point count colour:** red (<4), orange (4–6), green (≥7).

---

## Import / Export

- **Export:** Serialises `anchorFrames` to JSON and triggers a browser download.
- **Import:** Reads a JSON file, validates `anchorFrames` array, merges by `frame_idx` into the current anchor list (with a mismatch confirmation if indices differ).

---

## `drawCrosshair(ctx, cx, cy, color, label?)`

Draws a precision crosshair annotation marker:
1. Dark shadow lines for contrast on any background.
2. Coloured arm lines (length 7px from a 2px gap around the centre).
3. 2px filled circle at the centre.
4. Optional label in monospace 7px font with a semi-transparent background badge.

Earlier versions used a 5px filled circle, which was visually perceived as off-centre because the circle's area extends 5px in all directions from the click point. The crosshair design centres precisely on the clicked pixel.
