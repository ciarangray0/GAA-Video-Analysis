# Homography Module

`homography.py` computes the geometric mapping from camera image pixels to a fixed 2D pitch-canvas. It provides the core `compute_homographies_with_lines_v3` function (the v3 anchor computation) plus lower-level utilities used by other modules.

---

## Constants

| Constant | Value | Meaning |
|----------|-------|---------|
| `_REPROJECTION_OUTLIER_PX` | `30.0` | Error above which a keypoint is "outlier" |
| `_REPROJECTION_HIGH_PX` | `15.0` | Error above which a keypoint is "high" (warning) |

---

## Internal Helpers

### `_meters_to_canvas_pixels(x_m, y_m) → (float, float)`
Converts pitch coordinates in meters to canvas pixels:
```python
x_px = x_m / GAA_PITCH_WIDTH  * OUT_W   # = x_m * 10
y_px = y_m / GAA_PITCH_LENGTH * OUT_H   # = y_m * 10
```

### `_compute_coverage_score(pts_image, img_w, img_h, grid_cols=3, grid_rows=2) → float`
Measures spatial spread of annotated keypoints. Divides the image into a 3×2 grid (6 cells) and counts the fraction that contain at least one keypoint. Returns 0.0–1.0. A score below 0.5 triggers a "warning" quality label — annotations clustered in one region produce poorly conditioned homographies.

### `_hartley_normalize(pts) → (pts_normalized, T)`
Implements Hartley normalisation for DLT stability.

**Why this is mandatory:** Without normalisation, the DLT matrix `A` contains products of image coordinates (~0–1920) and canvas coordinates (~0–1400). These numbers can reach ~2×10⁶, and the subsequent SVD becomes numerically unstable — the singular vectors corresponding to the true solution get swamped by numerical noise, producing a wildly wrong H.

**Algorithm:**
1. Compute the centroid of the point set.
2. Shift all points so the centroid is at the origin.
3. Compute the mean distance from the origin.
4. Scale so the mean distance = √2.
5. Pack as a 3×3 similarity transform `T`:
   ```
   T = [[scale,   0,   -scale*cx],
        [  0,   scale, -scale*cy],
        [  0,     0,       1   ]]
   ```
6. Return normalised points and `T` (needed to denormalise H at the end).

The normalisation is independent for image coordinates and canvas coordinates (two separate calls to `_hartley_normalize`), producing `T_img` and `T_canvas`.

### `_compute_reprojection_errors(H, pts_image, pts_canvas) → np.ndarray`
For each point pair, applies `H` to the image point and measures Euclidean distance to the expected canvas point. Returns a 1D array of per-point errors in canvas pixels.

### `_fill_info(computation_info, frame_idx, H, keypoints, pts_image, pts_canvas, valid_lines, n_line_pts, img_width, img_height) → None`
Populates `computation_info[frame_idx]` with a quality report dict containing:
- `num_keypoints`, `keypoints` (per-point error + verdict)
- `repr_mean`, `repr_max` — mean and max reprojection error
- `coverage` — grid coverage score (or `None` if image dimensions unknown)
- `valid_lines`, `synthetic_points`
- `quality` — `"good"`, `"warning"`, or `"bad"` determined by:
  - "bad" if any outlier OR mean error > 30px
  - "warning" if mean error > 15px OR coverage < 0.5
  - "good" otherwise

---

## Public API

### `resolve_pitch_coordinates(pitch_id) → (x_m, y_m)`
Converts a `pitch_id` string to (x_meters, y_meters). Two formats are accepted:

1. **Named vertex:** looks up in `GAA_PITCH_VERTICES`. E.g. `"corner_tl"` → `(0.0, 0.0)`.
2. **Encoded line point:** matches regex `^line_.+_x([-\d.]+)_y([-\d.]+)$`. E.g. `"line_13m_top_x42.5_y13.0"` → `(42.5, 13.0)`. This format is used when the user clicks on a line segment in the pitch diagram (as opposed to a named vertex).

Raises `ValueError` for unrecognised IDs.

---

### `compute_homography(pts_image, pts_pitch_canvas) → np.ndarray`
Simple wrapper around `cv2.findHomography(RANSAC, threshold=5.0)`. Used for testing and as a fallback utility. The v3 endpoint uses `compute_homographies_with_lines_v3` instead.

---

### `map_pixel_to_pitch(x_img, y_img, H) → (x_canvas, y_canvas)`
Apply a homography to a single image pixel:
```python
p = [x_img, y_img, 1.0]
result = H @ p
result /= result[2]    # perspective divide
return result[0], result[1]
```

Called by `map_players_to_pitch` for every player detection.

---

## `compute_homographies_with_lines_v3(annotations, ...)`

### Signature
```python
def compute_homographies_with_lines_v3(
    annotations: Dict[int, Dict],
    num_samples_per_line: int = 10,
    ransac_iterations: int = 2000,
    ransac_threshold: float = 5.0,
    keypoint_weight: float = 20.0,
    img_width: Optional[int] = None,
    img_height: Optional[int] = None,
) -> Tuple[Dict[int, np.ndarray], Dict[int, dict]]
```

Returns `(homographies, computation_info)` where `homographies` maps frame_idx → 3×3 H and `computation_info` maps frame_idx → quality report dict.

### Step-by-Step Algorithm

#### Pre-check
Skip any frame with fewer than 4 keypoints — `cv2.findHomography` needs a minimum of 4 point correspondences. Records an error in `computation_info`.

#### Step 1 — Primary H from keypoints (RANSAC)

```python
H0, _ = cv2.findHomography(pts_image, pts_canvas, cv2.RANSAC, ransac_threshold, maxIters=ransac_iterations)
```

`H0` is the primary robust estimate. RANSAC identifies inliers among the annotated keypoints by iteratively fitting homographies to random 4-point subsets and keeping the one with the most points within `ransac_threshold` pixels. `maxIters=2000` gives a good chance of finding a clean solution even when some keypoints are misannotated.

If no line annotations are present, `H0` is used directly. This is the "keypoints-only" path.

#### Step 2 — Hartley-normalised weighted DLT

When line annotations are present, the system builds a combined linear system that incorporates both keypoints and line samples.

**Normalisation setup:**
```python
pts_image_n,  T_img    = _hartley_normalize(pts_image)
pts_canvas_n, T_canvas = _hartley_normalize(pts_canvas)
```
Only keypoint coordinates are used to compute the normalisation transforms. This ensures `T_img` and `T_canvas` capture the scale and centroid of the known-good point set.

**DLT matrix construction:**

For a homography `H` mapping image point `(u,v)` to canvas point `(x,y)`:
```
H @ [u, v, 1]ᵀ ∝ [x, y, 1]ᵀ
```

This gives two linear equations per point correspondence:
```
Row 1:  [u, v, 1,  0, 0, 0,  -x*u, -x*v, -x]  (enforces x-component)
Row 2:  [0, 0, 0,  u, v, 1,  -y*u, -y*v, -y]  (enforces y-component)
```

**Keypoint rows** — both rows added with weight `keypoint_weight` (default 20):
```python
rows.append([u, v, 1, 0, 0, 0, -x*u, -x*v, -x])
rows.append([0, 0, 0, u, v, 1, -y*u, -y*v, -y])
weights.extend([w_kp, w_kp])
```

**Line rows** — only one row per sample point (the constrained dimension), weight 1:

For a **horizontal line** (known `y_c` on canvas):
```python
# Only the Y-equation: [0, 0, 0, u, v, 1, -y_c*u, -y_c*v, -y_c]
rows.append([0, 0, 0, u, v, 1, -y_c*u, -y_c*v, -y_c])
weights.append(1.0)
```

For a **vertical sideline** (known `x_c` on canvas):
```python
# Only the X-equation: [u, v, 1, 0, 0, 0, -x_c*u, -x_c*v, -x_c]
rows.append([u, v, 1, 0, 0, 0, -x_c*u, -x_c*v, -x_c])
weights.append(1.0)
```

The canvas Y (or X) coordinate must also be normalised using `T_canvas`. Since `T_canvas` is a similarity transform, the normalised value is:
```python
y_c = scale_c * (y_c_raw - cy_c)
```
where `scale_c = T_canvas[0, 0]` and `cx_c, cy_c` are the centroid components.

**Why keypoint_weight = 20?**
With ~4 keypoints (contributing 8 rows at weight 20 = effective weight 160) and ~30 line samples (contributing 30 rows at weight 1 = effective weight 30), the ratio is roughly 5:1. Keypoints dominate the solution: they provide accurate 2D position. Lines can only gently pull the solution in directions the keypoints leave unconstrained (primarily X-skew in midfield).

#### Step 3 — Weighted SVD solve

```python
A = np.array(rows)          # shape (N, 9)
w_vec = np.array(weights)   # shape (N,)
_, _, Vt = np.linalg.svd(A * w_vec[:, np.newaxis], full_matrices=False)
H_norm = Vt[-1].reshape(3, 3)
```

The weighted DLT system `A·h = 0` (where `h` is the 9-element vector of H entries) is solved by finding the null vector of `A`. Multiplying each row by its weight is equivalent to solving a weighted least-squares system `min ||W·A·h||² subject to ||h||=1`. The solution is the right singular vector of `W·A` corresponding to the smallest singular value — always the last row of `Vt`.

#### Denormalisation

```python
H = np.linalg.inv(T_canvas) @ H_norm @ T_img
H /= H[2, 2]   # normalise so h[2,2]=1
```

The H in normalised space satisfies `T_canvas @ H @ T_img⁻¹ · p_n = q_n`. Re-expressed in original coordinates: `H = T_canvas⁻¹ @ H_norm @ T_img`.

#### Step 4 — Sanity check and fallback

```python
if np.any(np.isnan(H)) or np.linalg.cond(H) > 1e8 or _repr_mean(H) > _repr_mean(H0) * 2:
    H = H0.astype(np.float64)
```

Three failure modes trigger fallback to `H0`:
1. **NaN entries** — SVD produced an invalid result.
2. **Condition number > 10⁸** — H is nearly singular; applying it would produce extreme distortions.
3. **Reprojection doubled** — the line constraints pulled the solution away from the keypoints. This indicates the line annotations were inconsistent with the keypoints.

This fallback ensures that incorrect line annotations never produce a worse result than using keypoints alone.
