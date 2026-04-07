# Optical Flow Module

`constrained_homography.py` takes the small set of anchor homographies (one per user-annotated keyframe) and fills in a homography for every single frame in the video. It does this using optical flow — tracking how the pixels move between frames.

---

## Why per-frame propagation?

The user annotates maybe 5–10 anchor frames spread across the clip. Between those anchor frames, the camera is moving — panning, tilting, zooming. Without propagation, only those 5–10 frames would have a valid pitch mapping. Every other frame would be a blank.

Per-frame propagation answers the question: "Given that I know the camera position at frame 30 and frame 60, what is the camera position at frame 45?" It estimates the camera motion between consecutive frames using optical flow, then chains those small motion estimates together to cover the whole gap.

---

## Constants

| Constant | Value | Meaning |
|----------|-------|---------|
| `_LK_WIN_SIZE` | `(21, 21)` | The pixel neighbourhood examined around each tracked point |
| `_LK_MAX_LEVEL` | `3` | Number of image pyramid levels (see below) |
| `_LK_FB_THRESH` | `1.0 px` | Forward-backward consistency check threshold |
| `_LK_RANSAC_THRESH` | `3.0 px` | How close a point must be to be counted as an inlier |
| `sg_window_default` | `21` | Default window size for the final smoothing pass |
| `sg_order` | `2` | Polynomial degree used in smoothing |

---

## `_lk_inter_frame_H(g1, g2, mask, max_corners, corner_quality, min_distance)`

This function takes two consecutive greyscale frames (`g1` and `g2`) and returns a 3x3 homography `H` that describes how the camera moved between them.

Think of it like this: find some identifiable patches on the ground in frame `g1`, figure out where those exact same patches ended up in frame `g2`, then fit a homography to those before/after positions.

### Step 1 — Feature detection

```python
pts1 = cv2.goodFeaturesToTrack(g1, maxCorners, qualityLevel, minDistance, mask=mask)
```

Shi-Tomasi corner detection finds spots in `g1` that are "trackable" — places with strong texture in both the horizontal and vertical directions (actual corners or distinctive patches on the pitch surface). It returns up to `maxCorners` (default 500) such points.

The `mask` parameter tells the detector where it is allowed to look. The mask blacks out the top 35% of the image (sky, stadium stands, scoreboards). Those regions move independently of the camera and would confuse the tracker.

### Step 2 — Forward flow

```python
pts2, status, _ = cv2.calcOpticalFlowPyrLK(g1, g2, pts1, None, ...)
```

Lucas-Kanade (LK) optical flow tracks each point from `g1` into `g2`. It works by assuming a point's neighbourhood looks approximately the same in both frames and solving for the shift that best aligns the two patches. The `status` array marks each point as successfully tracked (1) or lost (0).

The "Pyr" in `calcOpticalFlowPyrLK` stands for pyramid — the algorithm first tracks on a small downscaled version of the image (coarse motion), then refines on higher resolution levels. `_LK_MAX_LEVEL = 3` means 4 levels total (original + 3 downscaled). This lets it handle large camera movements that would confuse a single-resolution tracker.

### Step 3 — Backward flow

```python
pts1_back, status_back, _ = cv2.calcOpticalFlowPyrLK(g2, g1, pts2, None, ...)
```

The same tracker is run in reverse: starting from where the points ended up in `g2`, track them back to `g1`. This gives us `pts1_back` — predicted original positions.

### Step 4 — Forward-backward consistency filter

```python
fb_error = np.linalg.norm(pts1 - pts1_back, axis=2).ravel()
good = fb_error < _LK_FB_THRESH  # _LK_FB_THRESH = 1.0 px
```

If the tracker is working well, `pts1_back` should be very close to the original `pts1`. The difference, `fb_error`, should be nearly zero for background points (the pitch, the lines) but will be large for moving players.

**Why does this catch players?** A player at position A in frame `g1` has moved to position B in frame `g2`. Forward tracking finds B correctly. But backward tracking from B goes back to where that player's patch came from — which is a different player or empty ground. So `pts1_back` ends up far from `pts1`, and the consistency check rejects the point.

This is an elegant self-check: points that pass are almost certainly static background. Points that fail are probably moving objects.

### Step 5 — Robust homography fitting

```python
H, inlier_mask = cv2.findHomography(pts1_good, pts2_good, cv2.RANSAC, _LK_RANSAC_THRESH)
```

With the surviving (background-only) point pairs, RANSAC fits a homography. This gives `H_{g1 → g2}` — the homography that maps image-space coordinates in frame `g1` to image-space coordinates in frame `g2`. It needs at least 8 inliers to return a valid result; below that it returns `None`.

Returns `(H, n_inliers)`, or `(None, 0)` on any failure.

---

## `build_optical_flow_per_frame_H(video_path, anchor_homographies, total_frames, ...)`

This is the main function. It takes the video file path, the anchor homographies (a dict mapping `frame_index → H`), and the total frame count. It returns a complete dict of `per_frame_H` covering every frame 0 to `total_frames - 1`.

The work happens in three phases.

---

### Phase 1 — Build optical-flow homographies for every consecutive pair

The video is decoded once, frame by frame. For each consecutive pair of frames `(t, t+1)`, `_lk_inter_frame_H` is called:

```
of_Hs[t] = H_{t → t+1}   for t = 0 .. total_frames - 2
```

Think of these as "step" homographies — each one says "if a pixel was at position P in frame t, the same point on the pitch would appear at position Q in frame t+1". Failed pairs (too few inliers) are stored in `failed_frames`.

The mask (top 35% blackout) is computed once and reused. Excluding the sky/stands prevents the tracker from latching onto fans waving flags or advertising boards that scroll — those would produce wildly wrong inter-frame Hs.

---

### Phase 2 — Chain and drift-correct per segment

Now we have one inter-frame step homography per frame pair. We need to chain these together, anchored to the trusted anchor Hs, to fill in every frame.

Consider a segment between anchor frame A and anchor frame B. We know `H[A]` exactly (it was computed directly from annotations). We want to fill in `H[A+1]`, `H[A+2]`, ..., `H[B-1]`.

**Forward chaining:**

```python
H[t] = H[t-1] @ inv(of_Hs[t-1])
```

Let's unpack this carefully. `of_Hs[t-1]` is `H_{(t-1) → t}` in image space — it maps an image-space point in frame `t-1` to where it appears in frame `t`. What we want is `H[t]`, which maps an image-space point in frame `t` to a canvas-space point.

Imagine standing at frame `t`. To get from image `t` to the canvas, you can:
1. Go backwards in image space to frame `t-1`: apply `inv(of_Hs[t-1])`.
2. Then use `H[t-1]` to map to the canvas.

So: `H[t] = H[t-1] @ inv(of_Hs[t-1])`.

If `of_Hs[t-1]` is None (the LK tracking failed for that pair), the previous H is reused for that frame — it is slightly wrong but much better than nothing.

**The drift problem:**

Each `of_Hs[t]` is a small estimate with small errors. When you chain 30 of them together, those errors accumulate — like a satnav that has been dead-reckoning for too long. By the time you reach anchor B, the chained estimate `H_chain[B]` may be noticeably different from the trusted `anchor_homographies[B]`.

**Drift correction:**

The drift is the gap between the chained estimate and the truth at anchor B:

```python
H_drift = anchor_homographies[B] @ inv(H_chain_B)
```

`H_drift` is the correction that, when applied after the chain, gives the correct anchor H. Rather than applying it all at once only at B (which would create a sudden jump), it is blended in gradually over the segment:

```python
alpha = (t - A) / (B - A)          # 0.0 at A, 1.0 at B
H_corr = (1 - alpha) * I + alpha * H_drift
H[t] = H_corr @ H_chain[t]
```

The `*` here is scalar multiplication of a matrix. `(1 - alpha) * I` is the identity matrix scaled by `(1 - alpha)`. So:
- At `t = A`: `alpha = 0.0`, `H_corr = I`, no correction at all.
- At `t = A + half-way`: `alpha = 0.5`, `H_corr` is halfway between the identity and `H_drift`.
- At `t = B`: `alpha = 1.0`, `H_corr = H_drift`, the full correction is applied.

This spreads the drift evenly across the segment so there is no visible jump.

After applying drift correction, both anchor frames are re-pinned to their exact known values:
```python
per_frame_H[A] = anchor_homographies[A]
per_frame_H[B] = anchor_homographies[B]
```

**Frames outside the annotated range:**

Frames before the first anchor are all assigned the first anchor's H. Frames after the last anchor are all assigned the last anchor's H. There is no optical flow information outside the annotated range.

---

### Phase 3 — Savitzky-Golay smoothing per segment

Even after drift correction, the 9 elements of each H matrix jitter slightly over time — small frame-to-frame noise from the optical flow estimates. Savitzky-Golay (SG) smoothing removes this.

**What SG smoothing does:** For each element of H (there are 9), it fits a polynomial of degree 2 to a sliding window of 21 frames, and replaces the centre frame's value with the value the polynomial predicts at that point. The effect is like a smooth curve drawn through noisy data.

Concretely, for a segment of frames [A, B]:

1. Collect all 9 H elements into a `(segment_length, 3, 3)` array.
2. Compute the effective window: `eff_window = min(21, segment_length)`, rounded down to the nearest odd number (SG requires an odd window).
3. If the window is less than 3, or the segment has fewer than 5 frames, skip smoothing — not enough data.
4. Apply `scipy.signal.savgol_filter(signal, eff_window, polyorder=2)` to each of the 9 scalar time series independently.
5. Renormalise each smoothed H so `H[2, 2] = 1` (this is the standard homography convention — the bottom-right entry should always be 1).
6. Re-pin the anchor frames exactly. SG smoothing might slightly alter the H values at exactly frames A and B, so we overwrite them with the trusted anchor values.

**Why smooth H elements directly?** Each element of H changes smoothly as the camera slowly pans or tilts. Smoothing those 9 numbers directly as time series is a good approximation for slow camera motion, and is much simpler than trying to decompose H into rotation/translation and smooth those.

---

## Return value: the `info` dict

The function returns a second value alongside `per_frame_H` — a diagnostic dictionary useful for debugging.

| Key | What it tells you |
|-----|-------------------|
| `num_frames` | Total frames processed |
| `failed_frames` | Frame indices where LK tracking produced too few inliers |
| `drift_at_anchors` | For each anchor, the Frobenius norm of the drift correction matrix |
| `corners_per_frame` | Number of LK inlier points used for each frame pair |
| `smoothing_window` | The default SG window that was used (21) |
| `unsmoothed_segments` | Segments that were too short to smooth |

**What is the Frobenius norm?** It is a single number measuring the "size" of a matrix — roughly, the square root of the sum of all its squared entries. A drift matrix that is close to the identity (no correction needed) will have a small Frobenius norm near 1. A large norm means the chain drifted significantly by that anchor — which suggests the anchor interval may be too long for this video.
