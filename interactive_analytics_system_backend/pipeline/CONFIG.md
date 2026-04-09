# Pipeline Configuration

Two files cooperate to define the numbers and geometry everything else in the pipeline depends on: `config.py` (runtime settings) and `gaa_pitch_config.py` (pitch geometry).

---

## Why have a central config at all?

If you write the number `850` in ten different files, and the canvas size ever needs to change, you have ten places to update — and you will miss at least one. Centralising constants means there is one source of truth. Any file that needs the canvas width imports `OUT_W` from config; it never writes the literal `850` itself.

---

## `config.py` — Runtime constants

| Constant | Value | Meaning |
|----------|-------|---------|
| `OUT_W` | `850` | Output canvas width in pixels |
| `OUT_H` | `1400` | Output canvas height in pixels |
| `YOLO_MODEL_PATH` | env var `YOLO_MODEL_PATH` or `"models/v8s_960_v9.pt"` | Path to the custom-trained YOLO weights |
| `DEFAULT_CONF` | `0.35` | Minimum confidence score for YOLO to report a detection |

### Why 850 wide and 1400 tall?

A real GAA pitch is 85 metres wide and 140 metres long. The canvas is exactly 10 pixels per metre in both directions:

```
850  pixels ÷ 85  metres = 10 px/m  (width)
1400 pixels ÷ 140 metres = 10 px/m  (height)
```

This makes coordinate conversion trivial. A player standing 23 metres from the left sideline is at x = 230 on the canvas. No division, no rounding, no floating point — just multiply by 10.

### What does `DEFAULT_CONF = 0.35` mean?

YOLO scores each detection from 0.0 (no confidence) to 1.0 (certain). A threshold of 0.35 means "only report detections that YOLO is at least 35% confident about". Setting this too low causes many false positives (blobs of crowd noise get reported as players). Setting it too high causes missed detections on partially occluded players. 0.35 is the value found to work best for this camera setup.

### What is `YOLO_MODEL_PATH` doing?

The pipeline reads the environment variable `YOLO_MODEL_PATH` first. If that variable is not set, it falls back to the string `"models/v8s_960_v9.pt"`. This pattern lets the server on a deployment machine point to a different location for the model file without changing any code — you just set the environment variable.

---

## `gaa_pitch_config.py` — Pitch geometry

### Pitch dimensions

```
GAA_PITCH_LENGTH = 140.0   (metres, runs top-to-bottom, the y-axis)
GAA_PITCH_WIDTH  = 85.0    (metres, runs left-to-right, the x-axis)
```

The y-axis points from the top goal down toward the bottom goal. The top of the canvas (y = 0) is one end of the pitch, the bottom (y = 1400 px = 140 m) is the other. Which physical goal is at the top depends entirely on which direction the camera faces — that is established by the user's annotations when they pick anchor points.

---

### `GAA_PITCH_VERTICES` — named landmark locations

This is a dictionary that maps human-readable names to real-world locations on the pitch, expressed in metres `(x, y)`. For example:

```
"corner_tl"       → (0, 0)         top-left corner of the pitch
"top_goal_lp"     → (39.25, 0)     left post of the top goal
"top_goal_rp"     → (45.75, 0)     right post of the top goal
"center_left"     → (0, 70)        left end of the halfway line
"center_right"    → (85, 70)       right end of the halfway line
"corner_br"       → (85, 140)      bottom-right corner
```

The goal posts sit 39.25 m and 45.75 m from the left sideline. The gap between them is 45.75 - 39.25 = 6.5 metres — the width of a GAA goal.

**How this dictionary is used:** When the user clicks on a point in the camera frame and says "this is the left goal post", they are saying that camera pixel maps to `(39.25, 0)` in metres, or equivalently `(392.5, 0)` in canvas pixels. The homography solver reads this lookup to find the destination coordinate.

---

### `GAA_PITCH_LINES` — horizontal lines across the pitch

Horizontal lines run the full width of the pitch (from x = 0 to x = 85 m). Each one is defined by a single y value in metres. These are used as line constraints in the homography computation — rather than needing to annotate individual points on a line, the user can mark a segment of a line and the solver treats the entire line as a single constraint.

| Line name | y (metres) | Physical line |
|-----------|-----------|---------------|
| `endline_top` | 0 | Top end line |
| `small_rectangle_top` | 4.5 | Top of the small rectangle (goal area) |
| `13m_top` | 13 | 13-metre line, top end |
| `20m_top` | 20 | 20-metre line, top end |
| `45m_top` | 45 | 45-metre line, top end |
| `65m_top` | 65 | 65-metre line, top end |
| `halfway` | 70 | Centre line |
| `65m_bottom` | 75 | 65-metre line, bottom end |
| `45m_bottom` | 95 | 45-metre line, bottom end |
| `20m_bottom` | 120 | 20-metre line, bottom end |
| `13m_bottom` | 127 | 13-metre line, bottom end |
| `small_rectangle_bottom` | 135.5 | Bottom of the small rectangle |
| `endline_bottom` | 140 | Bottom end line |

Note that there are two 45-metre lines — one 45 m from each end. The top one sits at y = 45 m. The bottom one sits at y = 140 - 45 = 95 m. Same for the 65 m and 20 m lines.

---

### `GAA_PITCH_SIDELINES` — vertical lines along the pitch

Vertical lines run the full length of the pitch (from y = 0 to y = 140 m). Each is defined by a single x value in metres.

| Line name | x (metres) | Physical line |
|-----------|-----------|---------------|
| `left_sideline` | 0 | Left boundary of the pitch |
| `right_sideline` | 85 | Right boundary |
| `13m_box_left` | 33 | Left side of the 13-metre box |
| `13m_box_right` | 52 | Right side of the 13-metre box |
| `small_arc_left` | 29.5 | Left side of the small arc |
| `small_arc_right` | 55.5 | Right side of the small arc |

The 13-metre box spans from x = 33 to x = 52, centred on the goal (which is at x = 39.25 to x = 45.75). The box is wider than the goal — it extends 6.5 m out to each side.

---

### Converting metres to canvas pixels

Both axes use the same formula: multiply by 10.

```python
x_canvas = x_metres * 10     (e.g. 39.25 m → 392.5 px)
y_canvas = y_metres * 10     (e.g. 70.0 m  → 700 px)
```

This is exact — `850 / 85 = 10` and `1400 / 140 = 10` with no remainder. Pitch vertices whose coordinates are multiples of 0.1 m will always land on exact tenth-pixel boundaries, and for the vast majority of lines and key points (which are integer or simple fraction metres), the canvas position is a whole number of pixels.

The formula written in full (as you will see it in the code) is:

```python
x_canvas = x_m / GAA_PITCH_WIDTH  * OUT_W    # same as x_m * 10
y_canvas = y_m / GAA_PITCH_LENGTH * OUT_H    # same as y_m * 10
```

This longer form is preferred in the code because it makes the intent clear — "scale from metres to canvas" — even if someone later changes `OUT_W` or `OUT_H` to something that is not 10x the pitch dimensions.
