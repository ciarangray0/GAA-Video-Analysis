# Team Classifier Module

`team_classifier.py` classifies each BotSort player track as `'ellistown'` or `'opposition'` by analysing the jersey colour in sampled video frames. Ellistown wear a distinctive orange-yellow jersey (OpenCV HSV hue ≈ 14–28), which sits in a clean gap below grass green (hue ≈ 35–40).

---

## Tunable Constants

All thresholds are defined at the top of `team_classifier.py` and can be adjusted without changing the algorithm logic.

| Constant | Value | Meaning |
|----------|-------|---------|
| `YELLOW_HUE_MIN` | `14` | Lower bound of Ellistown yellow in OpenCV H (0–179) |
| `YELLOW_HUE_MAX` | `28` | Upper bound of Ellistown yellow in OpenCV H |
| `YELLOW_SAT_MIN` | `100` | Minimum saturation for a pixel to count as yellow (excludes washed-out highlights) |
| `GREY_SAT_THRESHOLD` | `30` | Pixels with saturation below this are masked before analysis (glare, shadows, white lines) |
| `JERSEY_CROP_FRACTION` | `0.5` | Only the top half of the bounding box is inspected (jersey region — excludes legs, grass) |
| `MIN_YELLOW_FRACTION` | `0.15` | A track is Ellistown if ≥ 15% of non-grey jersey pixels are in the yellow range |

---

## `extract_jersey_yellow(frame, bbox) → (mean_hsv, yellow_fraction)`

Extracts the yellow pixel fraction from a single detection's jersey region.

**Steps:**
1. Clamp `bbox = (x1, y1, x2, y2)` to the frame dimensions.
2. Crop to the top `JERSEY_CROP_FRACTION` of the bounding box: `y2_crop = y1 + (y2 - y1) * 0.5`.
3. Convert the crop from BGR to HSV (`cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)`).
4. Build a saturation mask: keep only pixels where `S >= GREY_SAT_THRESHOLD`. Low-saturation pixels (white pitch markings, shadows, glare) are excluded.
5. If no pixels survive the mask, return `(zeros, 0.0)`.
6. Of the surviving pixels, count those where `H ∈ [YELLOW_HUE_MIN, YELLOW_HUE_MAX]` and `S >= YELLOW_SAT_MIN`.
7. Compute `yellow_fraction = n_yellow / n_masked`.

**Returns:**
- `mean_hsv` — `float32` array `[H, S, V]` averaged over all masked pixels. Used for the jersey-colour swatch in the frontend.
- `yellow_fraction` — float 0–1. The primary classification signal.

---

## `classify_tracks(video_path, detections, sample_frames=30) → Dict[int, dict]`

Classifies every player track in the detection list.

**Algorithm — single sequential video pass:**

The function avoids the naive approach of seeking to each sampled frame individually per track (which would decode the same frame once per track that samples it, and cause expensive backward seeks). Instead:

1. Group `detections` by `track_id`.
2. **Build a frame→samples map:** For each track, sort its detections by `frame_idx`, down-sample evenly (`step = max(1, len(dets) // sample_frames)`), then add each sampled detection to a dict keyed by `frame_idx`:
   ```
   frame_to_samples: Dict[int, List[(track_id, bbox)]]
   ```
   Multiple tracks that sample the same frame all appear in the same list entry — that frame is only decoded once regardless of how many tracks need it.
3. **Single forward pass:** Open the video with `cv2.VideoCapture`. Iterate `sorted(frame_to_samples)` — frame indices in ascending order so seeks are strictly non-decreasing. For each unique frame index: `cap.set(CAP_PROP_POS_FRAMES, frame_idx)`, read the frame once, then call `extract_jersey_yellow` for every `(track_id, bbox)` pair in that frame's entry.
4. Accumulate `mean_hsv` and `yellow_fraction` values per track in `track_hsv` / `track_yellow` dicts.
5. After the pass, classify each track:
   - Compute `mean_yellow = mean(yellow_fractions)`.
   - If `mean_yellow >= MIN_YELLOW_FRACTION` → `team = 'ellistown'`, `confidence = min(1.0, mean_yellow / (MIN_YELLOW_FRACTION * 2))`.
   - Otherwise → `team = 'opposition'`, `confidence = 1.0 - min(1.0, mean_yellow / MIN_YELLOW_FRACTION)`.
   - Store `{team, confidence: float (3 dp), mean_hsv: [H, S, V] (1 dp)}`.
6. Tracks with no usable samples (all frames unreadable) default to `'opposition'` with `confidence=0.0`.
7. Logs a summary: `"{N} ellistown, {M} opposition from {T} tracks"` at INFO level.

**Performance:** With N tracks all spanning the same clip, the naive approach performs up to `N × sample_frames` video decodes. The single-pass approach reduces this to the number of *unique* sampled frame indices. Since all tracks are present across the same clip, sampled frames overlap heavily across tracks, and the actual number of unique frames decoded is typically close to `sample_frames` regardless of track count.

**Returns:** `Dict[int, dict]` mapping `track_id → {team, confidence, mean_hsv}`.

**Note:** Referees should be filtered out before calling `classify_tracks` (use `filter_detections_for_mapping` in `map_players.py`). The function does not attempt to detect referees — they will be misclassified as `'opposition'` since referee kit is typically dark.

---

## `override_classification(classifications, track_id, new_team) → Dict[int, dict]`

Returns a new copy of `classifications` with `track_id` reassigned to `new_team`.

- If `track_id` exists: updates only the `team` field, preserving `confidence` and `mean_hsv`.
- If `track_id` does not exist: creates a new entry with `confidence=1.0` and `mean_hsv=[0,0,0]`.
- The input dict is never mutated — a shallow copy is returned.

Valid `new_team` values (enforced by the API layer, not by this function): `'ellistown'`, `'opposition'`, `'referee'`, `'ignore'`.

---

## API Integration

| Endpoint | Action |
|----------|--------|
| `POST /videos/{id}/classify-teams` | Calls `classify_tracks`; stores result in cache + disk |
| `GET /videos/{id}/classify-teams` | Returns stored classifications from cache or disk |
| `PATCH /videos/{id}/classify-teams` | Calls `override_classification`; updates cache + disk |

Classifications are persisted to `data/annotations/{id}_team_classifications.json` as `{str(track_id): {team, confidence, mean_hsv}}`.

---

## Design Notes

**Why top 50% of bbox?**
The lower half of a player bounding box often contains the legs (shorts/socks) and the ground (grass). Jersey colour is most reliably read from the torso. The 50% crop is a practical compromise — it includes the jersey while tolerating imprecise bounding box heights.

**Why saturation masking?**
GAA pitches have white line markings, and players' kit includes white numbers/stripes. Masking low-saturation pixels removes these without needing colour-specific exclusion rules.

**Why 15% yellow fraction threshold?**
Empirical calibration across two games. Ellistown yellow typically produces 25–50% yellow fractions; opposition (darker/non-yellow kit) produces 0–8%. The 15% midpoint leaves a comfortable margin. The confidence score encodes how far above or below the threshold the track sits.

**Confidence score formula:**
- Ellistown: `min(1.0, mean_yellow / (MIN_YELLOW_FRACTION * 2))` — reaches 1.0 at 30% yellow fraction (2× the threshold).
- Opposition: `1.0 - min(1.0, mean_yellow / MIN_YELLOW_FRACTION)` — reaches 0.0 at the threshold and 1.0 at 0% yellow.

Tracks with `confidence < 0.6` are reported in `low_confidence_tracks` by the POST endpoint summary.
