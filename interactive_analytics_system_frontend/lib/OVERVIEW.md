# `lib/` and `utils/` Overview

These folders contain helper code that is shared across multiple components. They are not React components themselves — they are plain TypeScript files with functions that components import and call. Think of them as a toolbox sitting next to the UI.

---

## `api.ts` — Talking to the Backend

Every time the frontend needs data from the backend, it goes through a function in `api.ts`. This centralises all network calls in one place so components don't each have their own `fetch()` calls scattered around.

### What is `fetch()`?

`fetch()` is the browser's built-in function for making HTTP requests. You give it a URL, it sends a request to a server, and eventually (asynchronously) gives you a response. Every function in `api.ts` is an `async` function — meaning it can pause and wait for the server to respond without freezing the rest of the app.

### Where does the URL come from?

```typescript
const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000"
```

`process.env.NEXT_PUBLIC_API_URL` is an environment variable — a value set outside the code, typically in a `.env` file. If it's not set, the code falls back to `http://localhost:8000`, which is where the backend runs during local development. The `NEXT_PUBLIC_` prefix is required by Next.js to make the variable available in the browser (not just on the server).

### How errors are handled

Every function checks whether the HTTP response was successful. If not, it reads the error message from FastAPI's standard `{ detail: "..." }` error format and throws a JavaScript `Error`. This means callers (the components) can use a simple `try/catch` block to handle failures and show the user a helpful message.

The one exception is `getTeamClassifications`, which returns an empty object `{}` on a 404 instead of throwing. This is intentional — a 404 just means no classification has been run yet, which is a normal state, not an error.

### API function reference

| Function | Method + Endpoint | What it does |
|---|---|---|
| `uploadVideo(file)` | `POST /videos` | Sends the MP4 file to the backend. Returns `VideoMetadata` (fps, frame count, dimensions, duration, and the video's ID that all other calls will use). |
| `trackVideo(videoId)` | `POST /videos/{id}/track` | Asks the backend to run YOLO object detection + BotSort tracking on the video. Returns frame count and the number of tracks found. |
| `getDetections(videoId)` | `GET /videos/{id}/detections` | Fetches all raw per-frame detection boxes (before mapping to the pitch). Throws on error. |
| `computeHomographies(videoId, annotations)` | `POST /videos/{id}/homographies/v3` | Sends all anchor frame annotations to the backend. The backend computes a homography matrix for each anchor frame (the math that maps the camera view to a top-down pitch view) and then propagates those matrices to every frame using optical flow. Returns a summary of frames processed. |
| `getAnchorQuality(videoId)` | `GET /videos/{id}/homographies/anchor-quality` | Fetches a quality report for each annotation keypoint — how accurately it was placed affects the homography accuracy. Used by `PipelineSteps` to display a quality table after step B runs. |
| `mapPlayers(videoId)` | `POST /videos/{id}/map_players` | Takes every player detection and projects it onto the 2D pitch using the per-frame homographies. Returns `PlayerPosition[]`. |
| `interpolateTrajectories(videoId, start, end, params)` | `POST /videos/{id}/interpolate?...` | Fills in the gaps between detected positions with smoothed interpolated positions. The `params` argument lets you tune the smoothing algorithm. Returns how many frames were generated. |
| `getPlayerPositions(videoId)` | `GET /videos/{id}/players` | Fetches all player positions (both detected and interpolated) for playback. |
| `classifyTeams(videoId)` | `POST /videos/{id}/classify-teams` | Analyses jersey colours to assign each player track to a team. Returns classifications and a summary. |
| `getTeamClassifications(videoId)` | `GET /videos/{id}/classify-teams` | Fetches stored classifications. Returns `{}` on 404 instead of throwing. |
| `overrideTeamClassification(videoId, trackId, team)` | `PATCH /videos/{id}/classify-teams` | Changes one player's team assignment manually. Returns the updated full classifications object. |
| `computeKpis(videoId, endFrame?)` | `POST /videos/{id}/compute-kpis` | Computes spatial KPIs (team spread, zones, centroids). The optional `endFrame` trims the analysis so trailing dead-ball frames don't skew the results. Returns a `KpiSummary`. |

### `InterpolationParams`

```typescript
interface InterpolationParams {
  sgLongWindow?: number  // Savitzky-Golay long smoothing window size
  sgMidWindow?: number   // Savitzky-Golay mid smoothing window size
  maxVelPx?: number      // Maximum allowed velocity in pixels per frame
}
```

The `?` on each field means it is optional. If you don't provide it, it is simply left out of the query string and the backend uses its own defaults.

---

## `pitch.ts` — Drawing the Pitch Canvas

This file contains all the functions that draw on HTML `<canvas>` elements. A canvas is like a blank painting surface — you draw on it imperatively by calling methods like `ctx.fillRect(...)` or `ctx.arc(...)` on the canvas's "2D context" (`ctx`).

There are two separate pitch views in the app: the **annotation pitch** (used during step 3 to pick landmark locations) and the **results pitch** (used during step 5 to show where players are). Each has its own drawing function.

### `pitchToCanvas(pitchX, pitchY)`

Converts a real-world pitch coordinate (in metres) to a pixel position on the display canvas.

The pitch is 85 metres wide and 140 metres long. The display canvas is 340 pixels wide and 560 pixels tall. So the scale factor is just:

```
pixel_x = (pitchX / 85.0) * 340   →   pitchX * 4
pixel_y = (pitchY / 140.0) * 560  →   pitchY * 4
```

So 1 metre = 4 pixels on screen. A player at the 45-metre line in the centre of the pitch would be at pixel (170, 180).

### `drawHorizontalLinePair(ctx, y1, y2, width)`

A small internal helper that draws two horizontal lines in a single `beginPath`/`stroke` call. Used for the symmetric yard lines (e.g. the 45m line at both ends is drawn as a pair). Grouping them into one call is more efficient than two separate `stroke()` calls, which each flush to the screen independently.

### `drawPitchDiagram(canvas, anchorFrames, currentAnchorIdx, pendingFrameClick, pendingLinePoint1)`

Draws the pitch diagram shown in the annotation UI. Here is what it draws, in order:

1. Sets the canvas size to 340×560 pixels, fills it solid green, adds a white border.
2. Draws the halfway line, then four pairs of symmetric horizontal lines (13m, 20m, 45m, and 65m lines) using `drawHorizontalLinePair`.
3. Draws the vertical lines of the 13-metre boxes on each end.
4. Draws the small goalkeeper boxes (the rectangle right in front of each goal).
5. If the user is in "pending frame click" mode (they clicked a line on the pitch diagram and now need to click the matching point on the video frame):
   - Highlights all clickable line segments in semi-transparent yellow.
   - Draws all known pitch vertices as yellow circles.
   - Shows a text overlay at the bottom with the pending pitch coordinates and instructions.
6. Otherwise: draws all vertices as orange circles (available to click).
7. Any vertices that have already been annotated for the current anchor frame are drawn in green.

### `drawPitch(canvas, positions, frame, teamClassifications?)`

Draws the results pitch — the live tactical view during playback. The `teamClassifications` argument is optional; if not provided, every player gets a colour based on their track ID.

Here is what it draws, in order:

1. Resets the canvas, fills it green, draws a slightly inset white border.
2. Draws all pitch line markings at 55% opacity (slightly transparent so player dots are clearly visible on top).
3. Draws 20-metre semicircles — the curved arc at each end of the pitch near the goal.
4. **Ghost dots**: players who appeared in earlier frames but are not present in the current frame are shown as grey semi-transparent dots at their last known position. This prevents the pitch from looking empty during tracking gaps. Ghost dots are drawn first so active dots appear on top.
5. **Active dots**: for each player at the current frame, draws a solid coloured circle (8px radius) with a white outline. If a player is out-of-bounds, the outline is red. The track ID is printed inside the dot.
6. A frame counter and player count overlay in the top-left corner.

### How player colours work

```typescript
function getPlayerColor(trackId, teamClassifications) {
  const classification = teamClassifications?.[trackId.toString()]
  if (classification?.team === 'referee') return null     // hide entirely
  if (classification?.team === 'ignore') return null      // hide entirely
  if (classification?.team === 'ellistown') return '#FFD700'   // gold
  if (classification?.team === 'opposition') return '#4488FF'  // blue
  // No classification — use the golden-angle colour scheme:
  return `hsl(${(trackId * 137.508) % 360}, 70%, 50%)`
}
```

Returning `null` means the player is not drawn at all — useful for the referee or tracking glitches you want to ignore.

The golden-angle colour scheme works by multiplying the track ID by 137.508 degrees (the "golden angle") and using that as a hue. Because 137.508 is irrational relative to 360, consecutive IDs get maximally different hues — track 1 and track 2 will never look similar even though they're numerically adjacent.

### `hsvToCss(h_cv, s_cv, v_cv)`

The backend stores jersey colours in OpenCV's HSV format, which uses unusual ranges: H is 0–179 (half the normal 0–360), and S/V are 0–255 (normally 0–100%). This function converts those values to a standard CSS `rgb()` string so they can be used in HTML elements. It is used by `ResultsViewer` to show jersey colour swatches in the team classification panel.

---

## `constants.ts` — Pitch Geometry Numbers

This file defines all the numbers that describe the pitch and the canvas. Centralising them here means you only need to change one number in one place if, for example, the display scale changes.

| Constant | Value | What it means |
|---|---|---|
| `PITCH_CANVAS_W` | `850` | Width of the backend's pitch canvas in pixels |
| `PITCH_CANVAS_H` | `1400` | Height of the backend's pitch canvas in pixels |
| `DISPLAY_SCALE` | `0.4` | How much we shrink the canvas for display (40% of 850 = 340) |
| `PITCH_DISPLAY_WIDTH` | `340` | Frontend canvas width in pixels (850 × 0.4) |
| `PITCH_DISPLAY_HEIGHT` | `560` | Frontend canvas height in pixels (1400 × 0.4) |
| `GAA_PITCH_WIDTH` | `85.0` | Real pitch width in metres |
| `GAA_PITCH_LENGTH` | `140.0` | Real pitch length in metres |
| `AVAILABLE_LINES` | Record | All named pitch lines with their positions in metres and whether they run horizontally or vertically |
| `GAA_PITCH_VERTICES` | Record | All named pitch landmarks (corners, goalpost bases, etc.) as `[x_metres, y_metres]` pairs |
| `PITCH_LINE_SEGMENTS` | Array | 17 named line segments covering every pitch marking, used for "click anywhere on this line" annotation |

### Why 850×1400 for the backend and 340×560 for the frontend?

The backend runs at 10 pixels per metre: 85m × 10 = 850px, 140m × 10 = 1400px. This gives good precision for the maths. But displaying an 850×1400 canvas directly on screen would be huge — so the frontend scales it down to 40% for display.

### `AVAILABLE_LINES` and `orientation`

Each line entry looks like this:

```typescript
'45m_top': { label: '45m Line (Top)', y_meters: 45.0 }
'left_sideline': { label: 'Left Sideline', x_meters: 0.0, orientation: 'vertical' }
```

Horizontal lines have a `y_meters` field. Vertical lines have `x_meters` and `orientation: 'vertical'`. The `AnchorFrameAnnotator` uses the `orientation` field to colour line annotations differently — cyan for horizontal, orange for vertical — so the user can easily tell them apart.

### `PITCH_LINE_SEGMENTS`

This is used when the user wants to annotate a point that lies somewhere along a line (not at a specific named vertex). For example, if a player is standing on the 45-metre line, the user can click any position along that line on the pitch diagram. The system encodes the exact pitch coordinates into the annotation's `pitch_id`, like `"line_45m_top_x42.5_y45.0"`.

---

## `utils/canvasUtils.ts` — Annotation Marker Drawing

### `drawCrosshair(ctx, cx, cy, color, label?)`

Draws the visual marker that appears on the video frame canvas when you annotate a keypoint. It is a small filled circle (2px radius) with four short crosshair arms extending outward — like the sight of a precision instrument.

The marker is drawn twice: once in dark colour (a "shadow" layer) and once in the target colour on top. This makes it visible against any background — white lines, grass, dark shadows — because the dark outline separates the coloured marker from whatever is behind it.

An optional `label` is drawn to the right of the crosshair with a semi-transparent background box, making it readable at any zoom level.

---

## `utils/formatters.ts` — Homography Quality Labels

After step B (homography computation) runs, the app shows a table of how accurately each annotation keypoint was placed. These helper functions turn raw numbers into coloured badges.

| Function | What it does |
|---|---|
| `reprErrorLabel(val)` | Takes a reprojection error in pixels and returns a string like `"15px ⚠"`. Under 10px is a pass, 10–20px is a warning, over 20px is a fail. |
| `reprErrorColor(val)` | Returns a CSS colour (green / amber / red) based on the same thresholds. |
| `qualityBadge(q)` | Turns an overall quality string (`"good"` / `"warning"` / `"bad"`) into a labelled emoji badge. |
| `qualityColor(q)` | Returns a CSS colour for an overall quality value. |
| `verdictBadge(v)` | Returns a short symbol (`"✓"` / `"⚠"` / `"✗"`) for a per-keypoint verdict. |
| `impactColor(impact)` | Returns a colour for whether a keypoint is `"helpful"`, `"marginal"`, or `"harmful"` to the homography. |

These are pure functions — they take a value and return a string or colour. They have no side effects and no dependencies on React.

---

## `utils/kpiUtils.ts` — KPI Data Processing

After the full pipeline runs and the user clicks "Compute KPIs", the backend returns a `KpiSummary` object with lots of raw numbers. These helpers process that data into formats that `ResultsViewer` can display cleanly.

### `ZONE_RANGES`

A simple lookup table for display strings:

```typescript
{ defensive: '0–47m', middle: '47–93m', attacking: '93–140m' }
```

Used to label zone charts without hardcoding the string everywhere.

### `teamColor(team)`

Returns a CSS colour for a team name. Gold for `'ellistown'`, blue for `'opposition'`, grey for anything else (unclassified). Consistent with the colours used on the results pitch canvas.

### `computeZoneAnalysis(kpiSummary)`

Takes the `zone_balance_timeseries` from the KPI summary (a list of per-frame zone counts for each team) and aggregates it across the whole clip.

For each frame, it sums up how many players each team had in each of the three pitch zones (defensive, middle, attacking). After looping through all frames, it returns:

- Cumulative player-frame counts per team per zone
- Combined zone activity totals (both teams added together)
- The `detectedZone` — the zone with the highest combined activity, which is used to determine whether the clip is an "attacking play" or "defensive play"

This is used by `ResultsViewer` to drive the zone bar charts and the "clip mode" label.

### `computeDepthSentence(spatialTimeseries, detectedZone)`

Produces one human-readable sentence describing how the team centroids moved during the clip. For example: `"Clip start: ellistown 12.3m goal-side · Clip end: ellistown 8.1m goal-side"`.

How it works:
1. Finds the first frame where both teams have a visible centroid (it skips frames where one team has no players on the pitch).
2. Finds the last such frame.
3. Computes the "goal-side gap" — the difference in centroid depth between the two teams — at both points.
4. Returns a string comparing the two.

If fewer than two frames have both teams present (e.g. it was a very short clip or classification failed), it returns `null` and the UI simply omits the sentence.
