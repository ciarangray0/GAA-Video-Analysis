# Optical Flow Module

`constrained_homography.py` propagates the sparse set of anchor homographies to every frame in the video using Lucas-Kanade optical flow, drift correction, and Savitzky-Golay smoothing.

---

## Why Per-Frame Propagation?

Anchor homographies are computed for user-selected keyframes (e.g. every 1 second). Between anchor frames, the camera is moving (PTZ pan/tilt/zoom, or a hand-held camera). Without propagation, only anchor frames would have valid pitch mappings. Per-frame propagation estimates the camera motion between consecutive frames and chains it to the nearest anchor's H.

---

## Constants

| Constant | Value | Meaning |
|----------|-------|---------|
| `_LK_WIN_SIZE` | `(21, 21)` | Pyramid window size for LK tracking |
| `_LK_MAX_LEVEL` | `3` | Number of image pyramid levels |
| `_LK_FB_THRESH` | `1.0` px | Forward-backward consistency error threshold |
| `_LK_RANSAC_THRESH` | `3.0` px | RANSAC threshold for inter-frame H fitting |
| `sg_window_default` | `21` | Default SG smoothing window per segment |
| `sg_order` | `2` | SG polynomial order |

---

## `_lk_inter_frame_H(g1, g2, mask, max_corners, corner_quality, min_distance)`

Computes a homography `H_{g1→g2}` from Lucas-Kanade optical flow.

### Steps:
1. **Feature detection:** `cv2.goodFeaturesToTrack(g1, maxCorners, qualityLevel, minDistance, mask=mask)` — finds Shi-Tomasi corners. The mask excludes the top fraction of the frame (sky/stands) so the tracked features are all ground-level.
2. **Forward flow:** `cv2.calcOpticalFlowPyrLK(g1, g2, pts1, ...)` — tracks each corner from frame `g1` to `g2`.
3. **Backward flow:** `cv2.calcOpticalFlowPyrLK(g2, g1, pts2, ...)` — tracks each found point back to `g1`.
4. **Forward-backward consistency filter:** A point is kept only if `||pts1 - pts1_back|| < _LK_FB_THRESH`. This removes moving players — their optical flow is inconsistent because they are not part of the rigid camera background.
5. **Robust H estimation:** `cv2.findHomography(src, dst, cv2.RANSAC, _LK_RANSAC_THRESH)` on the surviving points. Needs ≥ 8 inliers to return a valid H.
6. Returns `(H, n_inliers)`. Returns `(None, 0)` on any failure.

---

## `build_optical_flow_per_frame_H(video_path, anchor_homographies, total_frames, ...)`

### Signature
```python
def build_optical_flow_per_frame_H(
    video_path: str,
    anchor_homographies: Dict[int, np.ndarray],
    total_frames: int,
    max_corners: int = 500,
    corner_quality: float = 0.01,
    min_distance: float = 10.0,
    mask_top_fraction: float = 0.35,
) -> Tuple[Dict[int, np.ndarray], dict]
```

Returns `(per_frame_H, info)` where `per_frame_H` covers every frame 0..total_frames-1.

### Phase 1: Build Optical-Flow Homographies for All Consecutive Pairs

```
of_Hs[t] = H_{t→t+1}   for t = 0 .. total_frames-2
```

The video is decoded once, frame by frame, accumulating the grayscale image for each frame. For each consecutive pair `(t, t+1)`, `_lk_inter_frame_H` is called with the pre-computed mask (top 35% excluded). Failed pairs are recorded in `failed_frames`.

The mask is computed once (cached in `mask_cache`) as a binary image: `mask[top_rows:, :] = 255`. This prevents Shi-Tomasi from picking features in the sky, advertisements, or crowd that move independently of the camera.

### Phase 2: Chain and Drift-Correct Per Segment

Between each consecutive anchor pair `(A, B)`:

**Forward chaining:**
```python
H[t] = H[t-1] @ inv(of_Hs[t-1])   for t = A+1 .. B
```

This is the correct direction. `of_Hs[t-1]` maps frame `t-1` → frame `t` in image space. To go from `H[t-1]` (which maps image `t-1` → canvas) to `H[t]` (which maps image `t` → canvas), we need to compose: first apply the inverse of the camera motion (image space), then apply `H[t-1]`:
```
H[t] = H[t-1] @ inv(H_{t-1→t})
```

If `of_Hs[t-1]` is None (failed pair), the previous H is reused for that frame.

**Drift correction:**
Chaining accumulates small errors. At anchor `B`, the chained value `H_chain[B]` may differ from the trusted `anchor_homographies[B]`. The drift is:
```python
H_drift = anchor_homographies[B] @ inv(H_chain_B)
```
This is the correction needed to go from the chained estimate to the truth. It is applied with a linear blend over the segment:
```python
alpha = (t - A) / (B - A)
H_corr = (1 - alpha) * I + alpha * H_drift
H[t] = H_corr @ H_chain[t]
```
At `t = A`, `alpha = 0`, so no correction is applied (the chain starts from the trusted anchor). At `t = B`, `alpha = 1`, so the full correction is applied (the result matches the trusted anchor). The drift is spread linearly across the segment.

After drift correction, both anchor frames are re-pinned exactly:
```python
per_frame_H[A] = anchor_homographies[A]
per_frame_H[B] = anchor_homographies[B]
```

**Frames before the first anchor / after the last anchor:**
These frames are assigned the first/last anchor H directly (no optical flow available outside the annotated range).

### Phase 3: Savitzky-Golay Smoothing per Segment

Chaining + drift correction can still leave small jitter in each H element over time. SG smoothing removes this by fitting a polynomial locally to each time series.

For each segment `[A, B]`:
1. Collect the 9 H elements as a `(n_seg, 3, 3)` array.
2. Compute `eff_window = min(sg_window_default=21, n_seg)`, rounded down to an odd number.
3. If `eff_window < 3` or `n_seg < 5`, skip smoothing for this segment.
4. Apply `scipy.signal.savgol_filter(signal, eff_window, polyorder=2)` independently to each of the 9 H elements.
5. Normalise each smoothed H so `H[2,2] = 1`.
6. Re-pin anchor frames exactly (SG smoothing might slightly alter the anchor H values).

**Why smooth H elements directly?** Each element of H varies smoothly as the camera moves. Smoothing the 9 scalar time series is equivalent to smoothing the motion in parameter space, which is a good approximation for slow camera motion.

---

## Return Value: `info` Dict

| Key | Content |
|-----|---------|
| `num_frames` | Total frames processed |
| `failed_frames` | List of frame indices where LK flow failed |
| `drift_at_anchors` | Dict mapping anchor_idx → Frobenius norm of drift correction |
| `corners_per_frame` | Dict mapping frame → number of LK inliers for that pair |
| `smoothing_window` | The default SG window used (21) |
| `unsmoothed_segments` | List of (A, B) pairs too short for smoothing |

`drift_at_anchors` is logged at INFO level and helps diagnose large camera movements between anchor frames (high drift norm = the anchor interval may be too long for this video).
