# Homography Module

`homography.py` figures out the geometric mapping between "camera image pixels" and a fixed top-down pitch diagram. Think of it like answering the question: "If a player is standing at pixel (342, 891) in the camera frame, where does that correspond to on a flat 2D map of the pitch?"

The main function is `compute_homographies_with_lines_v3`. Everything else in the file is either a helper it calls internally, or a small utility used by other modules.

---

## What is a homography?

A homography is a 3x3 matrix `H`. When you multiply it by an image pixel coordinate, you get the corresponding canvas coordinate out. That's it.

```
image point  →  multiply by H  →  canvas point
```

In practice, image coordinates are things like (342, 891) — pixels on a camera frame. Canvas coordinates are things like (425, 630) — pixels on our 850x1400 top-down pitch diagram. The homography encodes the camera's position, rotation, and zoom in a single compact matrix.

One H per anchor frame is computed from the user's annotations. Every other frame gets an H derived from optical flow (see `OPTICAL_FLOW.md`).

---

## Constants

| Constant | Value | Meaning |
|----------|-------|---------|
| `_REPROJECTION_OUTLIER_PX` | `30.0` | If a keypoint's error is above this, it is flagged "outlier" (likely misannotated) |
| `_REPROJECTION_HIGH_PX` | `15.0` | If a keypoint's error is above this but below 30, it's flagged as a "high error" warning |

These thresholds are in canvas pixels. The canvas is 850x1400 px for an 85x140 m pitch, so 10 px/m. A 15 px error is 1.5 m — noticeable. A 30 px error is 3 m — clearly wrong.

---

## Internal Helpers

### `_meters_to_canvas_pixels(x_m, y_m)`

Converts pitch coordinates in metres to pixel coordinates on the canvas.

```python
x_px = x_m * 10
y_px = y_m * 10
```

The pitch is 85 m wide and 140 m long. The canvas is 850 px wide and 1400 px tall. So exactly 10 px per metre in both directions. The corner of the pitch (0 m, 0 m) maps to canvas pixel (0, 0). The centre of the pitch (42.5 m, 70 m) maps to canvas pixel (425, 700).

---

### `_compute_coverage_score(pts_image, img_w, img_h)`

Checks whether the user's annotated keypoints are spread out across the image, or all clustered in one corner.

Imagine dividing the camera frame into a 3-column by 2-row grid — 6 cells in total. This function counts how many of those 6 cells contain at least one keypoint, and returns that as a fraction. If 4 out of 6 cells are covered, the score is 4/6 = 0.67.

**Why does this matter?** A homography needs points from all over the image to be well-conditioned. If all your keypoints are in the top-left corner, the homography has no information about what happens at the bottom-right — it is just guessing. A coverage score below 0.5 triggers a "warning" quality label.

---

### `_hartley_normalize(pts)`

Rescales a set of points so they are centred at the origin and have a mean distance of about 1.41 (which is √2). Returns the rescaled points and the transform `T` that was applied.

**Why is this necessary?** The DLT algorithm (explained below) builds a matrix `A` where each row contains products like `x_image * x_canvas`. Image coords are roughly 0–1920, canvas coords are roughly 0–1400. Those products reach around 2,000,000. When you then run SVD on `A`, the huge numbers cause floating-point precision loss — the computer cannot distinguish small differences between very large numbers, and the result is garbage.

Hartley normalisation rescales both point sets to small numbers first. After normalisation, all coordinates are roughly in the range -2 to 2, the products are small, and SVD gives an accurate answer.

**The algorithm, step by step:**

1. Find the centroid (the average x and the average y) of all the points.
2. Subtract the centroid from every point so the centre of the point cloud is now at (0, 0).
3. Compute the mean distance from the origin across all points.
4. Divide every point by `(mean_distance / √2)` so the mean distance becomes exactly √2.
5. Pack this "shift then scale" operation into a 3x3 matrix `T`:

```
T = [[scale,   0,   -scale * cx],
     [  0,   scale, -scale * cy],
     [  0,     0,       1      ]]
```

Here `cx, cy` is the centroid and `scale` is the scale factor. Multiplying `T` by a point gives you the normalised version of that point.

6. Return the normalised points and `T`. We need `T` later to undo the normalisation.

This is done separately for image coordinates (producing `T_img`) and for canvas coordinates (producing `T_canvas`), because the two spaces have different scales and centroids.

---

### `_compute_reprojection_errors(H, pts_image, pts_canvas)`

Measures how accurate `H` is by testing it against the known point pairs.

For each pair of (image point, canvas point):
1. Apply `H` to the image point to predict where it should land on the canvas.
2. Measure the straight-line (Euclidean) distance between the predicted position and the actual known canvas position.

That distance, in canvas pixels, is the reprojection error for that point. A small error means the homography fits well. A large error means either the annotation was wrong or the homography is bad.

Returns an array of per-point errors, one number per keypoint.

---

### `_fill_info(...)`

Populates a quality report for a frame after its homography is computed. You can think of this as the "health check" function — it reads through the errors and assigns a verdict.

The verdict is stored in `computation_info[frame_idx]` and contains:
- **`repr_mean`** — the average reprojection error across all keypoints
- **`repr_max`** — the worst single-point error
- **`coverage`** — the coverage score from above
- **`quality`** — one of `"good"`, `"warning"`, or `"bad"`, determined by:
  - `"bad"` if any single point has error > 30 px, or if the mean error > 30 px
  - `"warning"` if the mean error > 15 px, or if the coverage score < 0.5
  - `"good"` otherwise

---

## Public API

### `resolve_pitch_coordinates(pitch_id)`

The frontend sends pitch point IDs as strings. This function converts them to (x metres, y metres). There are two formats:

1. **Named vertex** — a string like `"corner_tl"`. This is looked up in the `GAA_PITCH_VERTICES` dictionary. For example, `"corner_tl"` (top-left corner) gives `(0.0, 0.0)`.

2. **Encoded line point** — a string like `"line_13m_top_x42.5_y13.0"`. This is used when the user clicked on a line rather than a named point. The x and y values are embedded directly in the string and extracted with a regex (a pattern-matching rule). For example, `"line_13m_top_x42.5_y13.0"` gives `(42.5, 13.0)`.

If the ID doesn't match either format, a `ValueError` is raised.

---

### `map_pixel_to_pitch(x_img, y_img, H)`

Applies a homography to a single pixel coordinate to get the canvas position. This is called for every player detection in every frame.

```python
p = [x_img, y_img, 1.0]   # homogeneous coordinates (just append a 1)
result = H @ p             # matrix multiply: H times the point vector
result /= result[2]        # perspective divide (explained below)
return result[0], result[1]
```

**What is the `1.0` for?** Homogeneous coordinates are a standard trick that lets matrix multiplication represent perspective projection. You append a 1 to make a 2D point into a 3D vector, do the multiply, then divide by the third component to get back to 2D. That final division by `result[2]` is called the "perspective divide".

**Concrete example:** Suppose H maps pixel (500, 600) to canvas (300, 800). Then:
```
p = [500, 600, 1.0]
result = H @ p  →  [something like 600, 1600, 2.0]
result /= 2.0   →  [300, 800, 1.0]
return 300, 800
```

---

## `compute_homographies_with_lines_v3`

This is the main function. Given the user's annotations (keypoints and optional line annotations) for one or more anchor frames, it computes one H matrix per frame.

### Pre-check

Before doing anything, the function counts how many keypoints are annotated for the frame. `cv2.findHomography` needs at least 4 point pairs to work — a homography has 8 degrees of freedom (the 9th is fixed to 1), so 4 pairs give 8 equations, which is the minimum. Frames with fewer than 4 keypoints are skipped and marked as errors.

---

### Step 1 — Primary H from keypoints only (RANSAC)

```python
H0, _ = cv2.findHomography(pts_image, pts_canvas, cv2.RANSAC, ransac_threshold, maxIters=ransac_iterations)
```

RANSAC stands for "Random Sample Consensus". It is a way of fitting a model when some of your data points might be wrong.

Here is the intuition: you have, say, 8 annotated keypoints. Some of them might be slightly misplaced by the user. RANSAC works like this:
1. Pick 4 random keypoints.
2. Fit an H to those 4 points exactly.
3. Test that H against all 8 points. Count how many are within `ransac_threshold` pixels of where H predicts they should be. These are called "inliers".
4. Repeat 2000 times (`maxIters=2000`). Keep the H that had the most inliers.
5. Refit a final H using all inliers from the best round.

The result, `H0`, is the best robust estimate using only keypoints. If there are no line annotations, this is the final answer.

---

### Step 2 — Hartley-normalised weighted DLT with line constraints

When the user has also annotated lines, we can do better. We build a combined system of linear equations that incorporates both keypoints and line samples.

**What is DLT?** DLT stands for "Direct Linear Transform". It is a way of turning the problem of finding H into a system of linear equations that can be solved with standard maths. Here is the key idea:

A homography `H` maps image point `(u, v)` to canvas point `(x, y)`. Written out:
```
H @ [u, v, 1]  =  some_scale * [x, y, 1]
```

You can rearrange this to get two equations with H's entries on the left and zero on the right:
```
Row 1: [u, v, 1,  0, 0, 0,  -x*u, -x*v, -x]   (enforces the x-component)
Row 2: [0, 0, 0,  u, v, 1,  -y*u, -y*v, -y]   (enforces the y-component)
```

Those 9 numbers in brackets are one row of the matrix `A`. Each row "says" something about what H must be. Stack enough rows and you have a linear system.

**Normalisation step:**

Before building the rows, we normalise the coordinates:
```python
pts_image_n,  T_img    = _hartley_normalize(pts_image)
pts_canvas_n, T_canvas = _hartley_normalize(pts_canvas)
```

Only the keypoint coordinates are used to compute the scale and centroid for normalisation. This keeps the normalisation anchored to the reliable points.

**Building the matrix rows:**

For each **keypoint**, both equations are added — both the x-constraint and y-constraint. They are given a high weight (`keypoint_weight = 20`):

```python
rows.append([u, v, 1, 0, 0, 0, -x*u, -x*v, -x])   # x-constraint
rows.append([0, 0, 0, u, v, 1, -y*u, -y*v, -y])   # y-constraint
weights.extend([20.0, 20.0])
```

For each **line sample**, only the one constrained dimension is added — with weight 1:

- **Horizontal line** (we know the Y-coordinate on the canvas, but not X):
  ```python
  rows.append([0, 0, 0, u, v, 1, -y_c*u, -y_c*v, -y_c])  # y-constraint only
  weights.append(1.0)
  ```

- **Vertical line** (we know the X-coordinate on the canvas, but not Y):
  ```python
  rows.append([u, v, 1, 0, 0, 0, -x_c*u, -x_c*v, -x_c])  # x-constraint only
  weights.append(1.0)
  ```

**Why `keypoint_weight = 20`?**
Consider a typical frame with 6 keypoints and 3 annotated lines (10 samples each = 30 line points):
- Keypoints contribute 12 rows at weight 20 = effective weight 240
- Line samples contribute 30 rows at weight 1 = effective weight 30

The ratio is about 8:1 in favour of keypoints. Keypoints know both X and Y accurately. Line points only know one coordinate, and the user's two clicks might not be perfectly on the line. The weighting ensures lines gently improve the solution without overriding the keypoints.

---

### Step 3 — Weighted SVD solve

```python
A = np.array(rows)           # shape: (num_rows, 9)
w_vec = np.array(weights)    # shape: (num_rows,)
_, _, Vt = np.linalg.svd(A * w_vec[:, np.newaxis], full_matrices=False)
H_norm = Vt[-1].reshape(3, 3)
```

**What is SVD?** SVD (Singular Value Decomposition) is a fundamental operation in linear algebra — think of it as a very powerful generalisation of finding roots of an equation. Here it solves the system `A * h = 0` where `h` is the 9 entries of H stacked into a single vector.

The trick: the solution to `A * h = 0` (with the constraint that `||h|| = 1`) is the right singular vector of `A` corresponding to the smallest singular value. SVD returns its `Vt` matrix with rows sorted from largest to smallest singular value — so `Vt[-1]` (the last row) is always the solution we want.

Multiplying each row of `A` by its weight (`A * w_vec[:, np.newaxis]`) is equivalent to giving the heavier rows more influence over what the "best" solution looks like.

---

### Denormalisation

Because we normalised the coordinates before building `A`, the H we solved for is in normalised space. We need to undo that:

```python
H = np.linalg.inv(T_canvas) @ H_norm @ T_img
H /= H[2, 2]   # rescale so the bottom-right entry is 1
```

Think of it like working in centimetres instead of metres, then converting back. `T_img` maps original image coords to normalised ones. `T_canvas` maps original canvas coords to normalised ones. So to go from normalised H to original H: first undo the image normalisation (`@ T_img`), then undo the canvas normalisation (`@ inv(T_canvas)`).

---

### Step 4 — Sanity check and fallback to H0

After computing H with line constraints, we check three things:

```python
if np.any(np.isnan(H)) or np.linalg.cond(H) > 1e8 or _repr_mean(H) > _repr_mean(H0) * 2:
    H = H0
```

1. **NaN entries** — the SVD produced an invalid result (can happen if rows are all zeros or degenerate).
2. **Condition number > 10⁸** — the condition number measures how "sensitive" a matrix is. A high condition number means tiny input changes cause huge output changes. H is nearly singular — applying it would produce wildly wrong positions.
3. **Reprojection doubled** — the line constraints pulled the solution away from the keypoints, making the error more than twice as large. This means the line annotations were inconsistent with the keypoints (e.g., the user clicked in the wrong place on the pitch diagram).

In any of these cases, the function falls back to `H0` — the simpler keypoint-only RANSAC result. This guarantees that adding line annotations can never make things worse than not having them at all.
