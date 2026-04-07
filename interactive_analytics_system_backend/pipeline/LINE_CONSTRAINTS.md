# Line Constraints Module

`line_constraints.py` provides one key function — `sample_points_on_line` — plus re-exports the available line dictionaries from `gaa_pitch_config.py` so callers only need one import.

---

## What problem it solves

When the user annotates a frame, they click on recognisable landmarks — corner flags, goal posts, the intersection of lines. These produce "keypoints": pairs of (image pixel, pitch coordinate) where both the X and Y position on the pitch are known exactly. The DLT algorithm uses these to compute the homography.

The problem is that in midfield there are very few recognisable point intersections. The 45m line runs all the way across the pitch, but there is nothing distinctive at any particular point along it — no goal post, no corner flag. A camera pointed at midfield might have zero keypoints available.

However, the 45m line itself is clearly visible as a long white stripe across the image. If the user clicks two points on that stripe, we know something useful: every point on that stripe has the same Y-coordinate on the pitch (45 metres from the top endline). We do not know the X-coordinate — anywhere along the line is valid — but knowing Y is still a real constraint on the homography.

Line annotations capture exactly this. The user clicks two points that both lie somewhere on a known pitch line. `sample_points_on_line` turns those two clicks into 10 synthetic points spread evenly along the image-space line segment. Each synthetic point contributes a partial constraint (Y-only for horizontal lines, X-only for vertical lines) to the DLT system. See `HOMOGRAPHY.md` for how these constraints are encoded in the matrix rows.

---

## `get_available_lines()`

Returns a copy of `GAA_PITCH_LINES` — a dictionary mapping line name strings to their Y-coordinate in metres. This is called by the `/line-constraints/available-lines` endpoint so the frontend knows which lines can be annotated.

---

## `sample_points_on_line(u1, v1, u2, v2, num_samples=10)`

Takes two image-space pixel coordinates — `(u1, v1)` and `(u2, v2)` — representing two points the user clicked on a known pitch line. Returns 10 evenly-spaced pixel coordinates along the line segment between them.

### The algorithm, step by step

**Step 1 — Generate evenly spaced t-values:**

```python
t_values = np.linspace(0.0, 1.0, num_samples)
```

`np.linspace(start, stop, N)` creates N values from `start` to `stop`, evenly spaced. With `num_samples=10`:
```
t = [0.0, 0.111, 0.222, 0.333, 0.444, 0.556, 0.667, 0.778, 0.889, 1.0]
```

Think of `t` as "how far along the line segment are we?" — 0.0 means at the first click, 1.0 means at the second click, 0.5 means exactly halfway between them.

**Step 2 — Linear interpolation:**

```python
u_samples = (1 - t_values) * u1 + t_values * u2
v_samples = (1 - t_values) * v1 + t_values * v2
```

This is the standard **lerp** formula (linear interpolation). For a concrete example, suppose the user clicked `(u1=100, v1=200)` and `(u2=900, v2=400)`. At `t=0.5`:
```
u = (1 - 0.5) * 100 + 0.5 * 900 = 50 + 450 = 500
v = (1 - 0.5) * 200 + 0.5 * 400 = 100 + 200 = 300
```

So the midpoint sample is at pixel (500, 300) — exactly halfway between the two clicks. The formula works for any value of t.

Note that `*` here is element-wise multiplication — `t_values` is an array of 10 numbers, and NumPy applies the formula to all 10 at once without needing a loop.

**Step 3 — Return as an Nx2 array:**

```python
return np.stack([u_samples, v_samples], axis=1).astype(np.float32)
```

`np.stack(..., axis=1)` places `u_samples` and `v_samples` side by side to produce a 10x2 array. Each row is one point: `[u, v]`. `float32` is the numeric type — 32-bit floating point numbers, which is what OpenCV expects.

### Why 10 samples?

10 points per line gives a good balance in the DLT system. Consider a typical annotation with 6 keypoints and 3 line annotations (10 samples each = 30 line points):
- 6 keypoints × 2 equations × weight 20 = effective weight 240
- 30 line points × 1 equation × weight 1 = effective weight 30

The lines contribute about 1/8 of the total weight, enough to gently steer the solution in the constrained dimension without overriding the keypoints. The `num_samples_per_line` parameter in the v3 endpoint lets you adjust this if needed.

---

## Available lines

### Horizontal lines — `GAA_PITCH_LINES`

These lines run across the width of the pitch. Annotating them constrains the Y-coordinate. They are most useful in midfield where point intersections are rare.

```
"endline_top"     →  y =   0.0 m   (top endline)
"13m_top"         →  y =  13.0 m
"20m_top"         →  y =  20.0 m
"45m_top"         →  y =  45.0 m
"65m_top"         →  y =  65.0 m
"halfway"         →  y =  70.0 m   (centre of pitch)
"65m_bottom"      →  y =  75.0 m
"45m_bottom"      →  y =  95.0 m
"20m_bottom"      →  y = 120.0 m
"13m_bottom"      →  y = 127.0 m
"endline_bottom"  →  y = 140.0 m   (bottom endline)
```

Note that "top" and "bottom" are relative to the canvas coordinate system — `y = 0` is the top of the canvas and `y = 140` is the bottom.

### Vertical lines — `GAA_PITCH_SIDELINES`

These lines run along the length of the pitch. Annotating them constrains the X-coordinate. They are most useful when a sideline or the 13m box sides are clearly in shot.

```
"left_sideline"   →  x =  0.0 m
"right_sideline"  →  x = 85.0 m
"13m_box_left"    →  x = 33.0 m
"13m_box_right"   →  x = 52.0 m
"small_arc_left"  →  x = 29.5 m
"small_arc_right" →  x = 55.5 m
```

The small arc lines (29.5 m and 55.5 m) mark the edges of the small square near the goal. The 13m box lines (33 m and 52 m) mark the edges of the larger rectangle around the goal.
