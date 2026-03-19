# Pipeline Configuration

Constants and geometry definitions for the video analysis pipeline. Two files cooperate: `config.py` (runtime constants) and `gaa_pitch_config.py` (pitch geometry).

---

## `config.py`

| Constant | Value | Meaning |
|----------|-------|---------|
| `OUT_W` | `850` | Output canvas width in pixels |
| `OUT_H` | `1400` | Output canvas height in pixels |
| `YOLO_MODEL_PATH` | env `YOLO_MODEL_PATH` or `"models/v8s_960_v9.pt"` | Path to custom-trained YOLO model weights |
| `DEFAULT_CONF` | `0.35` | YOLO detection confidence threshold for local tracking |

### Why 850 × 1400?

A real GAA pitch is 85 m wide × 140 m long, so the canvas is exactly **10 px/m** in both dimensions. Converting between meters and canvas pixels is trivial multiplication / division by 10.

---

## `gaa_pitch_config.py`

### Pitch Dimensions

| Constant | Value |
|----------|-------|
| `GAA_PITCH_LENGTH` | `140.0` m (y-axis) |
| `GAA_PITCH_WIDTH` | `85.0` m (x-axis) |

The **y-axis points "away"** from the top goal toward the bottom goal. Frame 0 of any video may show either goal at the top — homography annotations establish the mapping.

---

### `GAA_PITCH_VERTICES`

A dictionary of named pitch feature locations `(x_meters, y_meters)`. Used to look up destination points when the user annotates a keypoint with a `pitch_id`.

Key groups:

| Group | Example keys |
|-------|-------------|
| Corners | `corner_tl`, `corner_tr`, `corner_bl`, `corner_br` |
| Goal posts | `top_goal_lp` (39.25, 0), `top_goal_rp` (45.75, 0), `bottom_goal_lp/rp` |
| Goalie box | `left_box_top` (35.5, 4.5), `right_box_top` (49.5, 4.5), `*_bottom` |
| 13m box | `left_13m_box_top` (33, 13), `right_13m_box_top` (52, 13), `*_bottom` + endline variants |
| Small arc | `left_small_arc_top` (29.5, 20), `right_small_arc_top` (55.5, 20), etc. |
| Yard lines | `left_13m_line_top/bottom`, `left_20m_line_top/bottom`, etc. at x=0 and x=85 |
| Halfway | `center_left` (0, 70), `center_right` (85, 70) |

---

### `GAA_PITCH_LINES`

Horizontal lines crossing the full pitch width. Used as DLT line constraints (1D Y-constraint per sample point).

| Key | Y (meters) | Real line |
|-----|------------|-----------|
| `endline_top` | 0 | Top endline |
| `small_rectangle_top` | 4.5 | Top goal area |
| `13m_top` | 13 | 13m line (top) |
| `20m_top` | 20 | 20m line (top) |
| `45m_top` | 45 | 45m line (top) |
| `65m_top` | 65 | 65m line (top) |
| `halfway` | 70 | Centre line |
| `65m_bottom` | 75 | 65m line (bottom) |
| `45m_bottom` | 95 | 45m line (bottom) |
| `20m_bottom` | 120 | 20m line (bottom) |
| `13m_bottom` | 127 | 13m line (bottom) |
| `small_rectangle_bottom` | 135.5 | Bottom goal area |
| `endline_bottom` | 140 | Bottom endline |

---

### `GAA_PITCH_SIDELINES`

Vertical lines running the full pitch length. Used as DLT line constraints (1D X-constraint per sample point).

| Key | X (meters) | Real line |
|-----|------------|-----------|
| `left_sideline` | 0 | Left boundary |
| `right_sideline` | 85 | Right boundary |
| `13m_box_left` | 33 | Left 13m box side |
| `13m_box_right` | 52 | Right 13m box side |
| `small_arc_left` | 29.5 | Left small arc |
| `small_arc_right` | 55.5 | Right small arc |

---

### Meters-to-Canvas Conversion

Both `homography.py` and `app.py` use the same formula:

```python
x_canvas = x_m / GAA_PITCH_WIDTH  * OUT_W   # = x_m * 10
y_canvas = y_m / GAA_PITCH_LENGTH * OUT_H   # = y_m * 10
```

Because `OUT_W / GAA_PITCH_WIDTH = 850 / 85 = 10` and `OUT_H / GAA_PITCH_LENGTH = 1400 / 140 = 10`, this is always exact integer arithmetic for vertices whose coordinates are multiples of 0.1 m.
