# Line Constraints Module

`line_constraints.py` provides a point-sampling utility for the homography line constraint system. It also re-exports `GAA_PITCH_LINES` and `GAA_PITCH_SIDELINES` from `gaa_pitch_config.py` for convenience, so callers only need one import.

---

## Purpose

Keypoints provide **full 2D correspondences** (both X and Y are known on the pitch). But in midfield regions — far from goal posts and corner flags — there are no easily identifiable point intersections visible in the camera image. However, horizontal yard lines (13m, 20m, etc.) and sidelines **are** visible as streaks across the image.

A line annotation lets the user click two points that both lie on a known pitch line. The system then generates `N` synthetic correspondences along that segment. Each synthetic point provides a **1D constraint** rather than a 2D one:
- **Horizontal line** (e.g. 45m_top): the Y-coordinate on the pitch canvas is known, but the X-coordinate is free.
- **Vertical line** (e.g. left_sideline): the X-coordinate on the pitch canvas is known, but the Y-coordinate is free.

This constraint is expressed as a single row in the DLT matrix (see `HOMOGRAPHY.md` for the DLT formulation).

---

## `get_available_lines() → Dict[str, float]`
Returns a copy of `GAA_PITCH_LINES` (horizontal line IDs → Y meters). Used by the `/line-constraints/available-lines` endpoint to tell the frontend what lines can be annotated.

---

## `sample_points_on_line(u1, v1, u2, v2, num_samples=10) → np.ndarray`
Uniformly samples `num_samples` points along the image-space line segment from `(u1, v1)` to `(u2, v2)`.

**Algorithm:**
1. Generates `t_values = np.linspace(0.0, 1.0, num_samples)`.
2. Computes `u = (1-t)*u1 + t*u2`, `v = (1-t)*v1 + t*v2` for each t.
3. Returns as `float32` Nx2 array.

The first and last points are the exact endpoints (t=0, t=1). All intermediate points are equally spaced. The minimum is `num_samples=2` (just the two endpoints).

**Why 10 samples?** Empirically, 10 points per line provides a good balance: enough to constrain the homography without over-weighting the line relative to the keypoints. The `num_samples_per_line` parameter in the v3 endpoint lets the user adjust this.

---

## Available Lines (from `gaa_pitch_config.py`)

### Horizontal lines — `GAA_PITCH_LINES`
Constrain Y-position. Useful in midfield and around the 45m/65m lines where point intersections are rare.

```python
"endline_top":    0.0   "13m_top":  13.0   "20m_top":  20.0
"45m_top":       45.0   "65m_top":  65.0   "halfway":  70.0
"65m_bottom":    75.0   "45m_bottom": 95.0  "20m_bottom": 120.0
"13m_bottom":   127.0   "endline_bottom": 140.0
```

### Vertical lines — `GAA_PITCH_SIDELINES`
Constrain X-position. Useful when the sideline or 13m box sides are clearly visible.

```python
"left_sideline":  0.0    "right_sideline": 85.0
"13m_box_left":  33.0    "13m_box_right":  52.0
"small_arc_left": 29.5   "small_arc_right": 55.5
```
