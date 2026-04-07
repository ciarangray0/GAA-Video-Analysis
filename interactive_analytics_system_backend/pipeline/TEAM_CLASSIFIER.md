# Team Classifier Module (`team_classifier.py`)

This module looks at each player's bounding box across a sample of video frames and figures out which team they play for — Ellistown or the opposition — based purely on jersey colour. Ellistown wear a distinctive orange-yellow jersey, which makes this feasible with a colour filter rather than a full machine-learning classifier.

---

## The core idea

In OpenCV, colours are represented in **HSV** format: Hue (the colour itself, e.g. red, yellow, green), Saturation (how vivid the colour is, 0 = grey, 255 = fully saturated), and Value (brightness). This is more useful than RGB for colour filtering because orange-yellow is always orange-yellow regardless of how bright or dark the lighting is — the Hue stays roughly the same.

OpenCV's Hue range runs from 0 to 179 (not 0 to 360 as you might expect from a colour wheel). Ellistown's orange-yellow sits at roughly Hue 14–28 on this scale. Grass green starts around Hue 35, so there is a clean gap between them.

---

## Tunable constants

All thresholds are defined at the top of `team_classifier.py`. Change the values here rather than inside the algorithm logic.

| Constant | Value | What it means |
|----------|-------|---------------|
| `YELLOW_HUE_MIN` | `14` | The lower Hue boundary for Ellistown yellow |
| `YELLOW_HUE_MAX` | `28` | The upper Hue boundary for Ellistown yellow |
| `YELLOW_SAT_MIN` | `100` | A pixel must be at least this saturated to count as yellow (rules out washed-out white/grey highlights that happen to fall in the Hue range) |
| `GREY_SAT_THRESHOLD` | `30` | Pixels below this saturation level are completely ignored before analysis — they represent white pitch markings, glare, and dark shadows, none of which are jersey colour |
| `JERSEY_CROP_FRACTION` | `0.5` | Only the top half of the bounding box is inspected |
| `MIN_YELLOW_FRACTION` | `0.15` | If at least 15% of the valid jersey pixels are yellow, the player is classified as Ellistown |

---

## `extract_jersey_yellow(frame, bbox) → (mean_hsv, yellow_fraction)`

**What it does:** Takes one video frame and one player's bounding box, and returns how "yellow" that player's jersey region is.

**Step by step:**

1. **Clamp the bounding box** to the frame edges. Sometimes the tracker predicts a box that goes slightly off-screen; clamping prevents an array index error when cropping.

2. **Crop to the jersey region.** The full bounding box includes the player's legs, shorts, socks, and the grass underneath. Only the top half is kept — this is the torso where the jersey colour is most visible. If the box runs from y=100 to y=200, the crop is y=100 to y=150.

3. **Convert from BGR to HSV.** OpenCV loads images in Blue-Green-Red order by default. `cv2.cvtColor` converts the crop to the Hue-Saturation-Value format described above.

4. **Build a saturation mask.** Create a binary mask that is `True` for every pixel where Saturation ≥ `GREY_SAT_THRESHOLD` (30). Pixels below this threshold are grey, white, or near-black — not jersey colour. They are excluded from all further calculations.

5. **Early exit.** If no pixels survive the saturation mask (e.g. the box is fully in shadow), return zeroes immediately.

6. **Count yellow pixels.** Among the surviving (non-grey) pixels, count how many have Hue between 14 and 28 **and** Saturation ≥ 100. These are the yellow pixels.

7. **Compute the fraction.** `yellow_fraction = number_of_yellow_pixels / number_of_non_grey_pixels`. A value of 0.3 means 30% of the visible jersey pixels are in the yellow range.

**Returns:**
- `mean_hsv` — the average [H, S, V] values across all non-grey pixels. This is stored and shown as a colour swatch in the frontend so you can visually check what colour the classifier saw.
- `yellow_fraction` — the number used for classification. Higher = more yellow.

---

## `classify_tracks(video_path, detections, sample_frames=30) → dict`

**What it does:** Classifies every player track in the detection list as `'ellistown'` or `'opposition'`.

**The performance problem it solves:**

The naive approach would be: for each track, pick 30 sample frames, seek to each one, read it, and call `extract_jersey_yellow`. If there are 20 tracks and 30 samples each, that is 600 frame reads — and many of those frames are the same frame read 20 times (once per track that appears in it). Seeking backwards in a video file is also slow.

The smarter approach: build a map of which frames are needed, then read each frame exactly once in forward order.

**Step by step:**

1. **Group detections by track.** Each group is one player's full list of bounding boxes across the whole clip.

2. **Build the frame-to-samples map.** For each track, sort its detections by frame number, then pick evenly spaced samples (e.g. every 5th detection if there are 150 detections and `sample_frames=30`). For each sampled detection, add an entry to a dict: `{frame_index: [(track_id, bbox), (track_id, bbox), ...]}`. Multiple tracks that need the same frame appear in the same list.

3. **Single forward pass through the video.** Open the video file once. Sort the frame indices in ascending order (so the video head only moves forward). For each frame index:
   - Seek to that position with `cap.set(CAP_PROP_POS_FRAMES, frame_idx)`.
   - Read the frame once.
   - Call `extract_jersey_yellow` for every `(track_id, bbox)` pair that requested this frame.
   - Accumulate the `yellow_fraction` values per track.

4. **Classify each track.** After the pass, each track has a list of `yellow_fraction` values — one per sampled frame. Compute the mean. Then:
   - If `mean_yellow >= 0.15` (the `MIN_YELLOW_FRACTION` threshold): team = `'ellistown'`. Confidence = `min(1.0, mean_yellow / 0.30)`. At 30% yellow the confidence reaches 1.0; at exactly 15% it is 0.5.
   - If `mean_yellow < 0.15`: team = `'opposition'`. Confidence = `1.0 - min(1.0, mean_yellow / 0.15)`. At 0% yellow the confidence is 1.0; at the 15% threshold it drops to 0.0.

5. **Handle unreadable tracks.** If a track had no usable frames (e.g. all frames were unreadable), it defaults to `'opposition'` with `confidence=0.0` — a safe fallback that signals low certainty.

6. **Log a summary.** Logs a line like `"12 ellistown, 8 opposition from 20 tracks"` so you can sanity-check the result at a glance.

**Returns:** `{track_id: {"team": str, "confidence": float, "mean_hsv": [H, S, V]}}`.

**Important note:** Referees should be filtered out before calling this function. The classifier only knows about orange-yellow vs everything else. Referee kit (typically dark) will be classified as `'opposition'`, which is incorrect. Use `filter_detections_for_mapping` in `map_players.py` to remove referee detections first.

---

## `override_classification(classifications, track_id, new_team) → dict`

**What it does:** Returns a corrected copy of the classifications dict with one track reassigned to a different team.

- If the track already exists, only the `team` field is updated. The `confidence` and `mean_hsv` values are left unchanged.
- If the track does not exist in the dict (unusual), a new entry is created with `confidence=1.0` and `mean_hsv=[0,0,0]`.
- The original dict is **never modified** — a new copy is returned. This is safer: if something goes wrong, the original is still intact.

Valid team strings (enforced by the API layer): `'ellistown'`, `'opposition'`, `'referee'`, `'ignore'`.

---

## API integration

| Endpoint | What it does |
|----------|--------------|
| `POST /videos/{id}/classify-teams` | Runs `classify_tracks`; saves the result to disk and the in-memory cache |
| `GET /videos/{id}/classify-teams` | Returns the stored classifications |
| `PATCH /videos/{id}/classify-teams` | Calls `override_classification` for one track; updates disk and cache |

Classifications are stored on disk at `data/annotations/{id}_team_classifications.json`. The JSON uses string keys (as all JSON keys must be strings), so the format is `{"42": {"team": "ellistown", "confidence": 0.87, "mean_hsv": [21.0, 142.0, 190.0]}, ...}`.

---

## Design decisions explained

**Why crop to the top 50% of the bounding box?**
The bottom half of a player's bounding box almost always contains shorts, socks, and grass — none of which reliably reflect jersey colour. The 50% line is a practical cutoff: it consistently captures the jersey torso while keeping the implementation simple. A more sophisticated approach would try to detect the jersey region specifically, but that would require another model.

**Why mask out low-saturation pixels?**
GAA pitches have white boundary lines and players' jerseys often have white numbers or stripes. Without saturation masking, these white areas would count as "not yellow" and dilute the fraction — but they are not actually jersey colour at all. Masking them out means `yellow_fraction` reflects the ratio of coloured-jersey pixels only.

**Why 15% as the threshold?**
This was calibrated empirically across two games. Ellistown yellow consistently produced fractions of 25–50%. Non-yellow kits consistently produced 0–8%. The 15% midpoint sits in the empty gap between those two ranges, giving a comfortable margin on either side for variation in lighting, camera angle, and partial occlusion.

**Why do confidence scores use 2× and 1× the threshold?**
The confidence formula maps the yellow fraction onto a 0–1 scale, with 1.0 meaning "very confident". For Ellistown, the scale reaches 1.0 at 30% yellow (2× the threshold), because that is a reliably high yellow fraction. For opposition, the scale bottoms out at 0.0 at the 15% threshold — meaning the track is at the borderline — and rises to 1.0 at 0% yellow. Tracks with `confidence < 0.6` are flagged in the POST endpoint response as potentially unreliable.
