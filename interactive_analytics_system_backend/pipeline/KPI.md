# KPI Module (`kpi.py`)

This module answers the coaching question: **where were the players and how did space open up during the clip?** It takes the list of mapped player positions (already converted to pitch coordinates) and produces statistics like distance covered, team shape, and how many players were in each zone of the pitch.

Speed-based metrics (sprints, high-speed running) are intentionally left out. On a short 10–30 second clip, they are dominated by noise from the camera-to-pitch projection, not real player movement.

---

## Key constants

Before diving into the functions, there are a few numbers defined at the top of the file that everything else depends on.

| Constant | Value | What it means |
|----------|-------|---------------|
| `PX_PER_METRE` | `10.0` | The pitch canvas is 850×1400 pixels representing an 85×140 m pitch — so 10 pixels = 1 metre |
| `PITCH_LENGTH_M` | `140.0` | The full pitch is 140 metres long (the y-axis) |
| `THIRD_1_END_M` | `46.667` | Where the first third ends: 140 ÷ 3 ≈ 46.7 m |
| `THIRD_2_END_M` | `93.333` | Where the second third ends: 140 × 2/3 ≈ 93.3 m |

The pitch is cut into three horizontal bands called "thirds". Think of it like dividing a football pitch into three equal strips from one goal to the other. These are called `defensive`, `middle`, and `attacking` in the code — but those labels are **absolute** (based on y position), not relative to which way Ellistown is playing. The frontend handles the flip.

---

## Helper functions

### `_get(p, key)`

A convenience function that reads a named field from `p`, whether `p` is a plain Python dictionary (e.g. `p["x_pitch"]`) or a Pydantic object (e.g. `p.x_pitch`). This lets the rest of the module not care which format it receives.

### `_team_label(cls) → str`

Pulls the `team` string (like `"ellistown"` or `"opposition"`) out of a classification dict or object. If `cls` is `None` (the player was never classified), returns the string `"unclassified"` as a safe fallback.

---

## `compute_player_distances(positions) → dict`

**What it does:** For every player track in the clip, adds up how far they moved in total, in metres.

**Think of it like this:** Imagine you plotted a player's position as a dot on the pitch at every frame. This function connects all those dots with straight lines and adds up the length of each line segment.

**Step by step:**

1. Group all positions by `track_id`. Each group is one player's set of (frame, x, y) entries.

2. Sort each group by `frame_idx` — so the positions are in time order, not random order.

3. For each consecutive pair of positions, compute the change in x and y (called `dx` and `dy`). Divide by `PX_PER_METRE` (which is 10) to convert from pixels to metres. So if a player moved 50 pixels to the right, that is 5 metres.

4. Use the Pythagorean theorem to get the straight-line distance between each pair: `sqrt(dx² + dy²)`. For example, if `dx = 3 m` and `dy = 4 m`, the distance is `sqrt(9 + 16) = 5 m`.

5. Add all those segment lengths together to get `total_distance_m`.

**A note on frame gaps:** If the tracker lost a player for 10 frames and then found them again, there will be a gap in the positions. The code still handles this correctly — it just adds a longer "step" for that gap. It doesn't try to invent positions for the missing frames.

**Returns:** A dictionary like `{42: {"total_distance_m": 18.3}, 7: {"total_distance_m": 22.1}, ...}` where each key is a `track_id`.

---

## `compute_team_spatial(positions, team_assignments, frame_idx) → dict`

**What it does:** Takes a snapshot of where every player is at one specific frame, groups them by team, and computes two things per team: where is the team's "centre of gravity", and how spread out are they?

**Step by step:**

1. Filter `positions` down to only the entries where `frame_idx` matches the requested frame.

2. Group players by team using `team_assignments` (a dict mapping `track_id` → team). Players labelled `'referee'` or `'ignore'` are skipped.

3. Convert each player's pixel coordinates to metres by dividing by 10. A player at `(350 px, 700 px)` is at `(35 m, 70 m)` on the real pitch.

4. **Centroid:** Compute the average x and average y for all players on each team. If Ellistown's five visible players are at x positions 20, 25, 30, 35, 40 m, their centroid x is `(20+25+30+35+40) / 5 = 30 m`. This is the "centre of gravity" of the team.

5. **Spread (convex hull area):** If SciPy is installed and the team has at least 3 players, compute the convex hull. Think of it like stretching a rubber band around all the player dots — the area inside that rubber band is `spread_m2`. A compact, organised defensive unit will have a small area; a stretched attacking team will have a large one. If the players happen to be in a straight line, the hull has zero area and the code falls back to `0.0`.

6. **Centroid separation:** If both teams have players visible, compute the straight-line distance between Ellistown's centroid and the opposition's centroid. A small separation means both teams are compressed into the same part of the pitch; a large separation means the play is stretched.

**Returns:** A nested dict with centroid, spread, and player count per team, plus the centroid separation.

---

## `compute_zone_balance(positions, team_assignments, frame_idx) → dict`

**What it does:** For one frame, counts how many players each team has in each of the three pitch thirds.

**The thirds:**

| Zone name | y range on the pitch |
|-----------|---------------------|
| `defensive` | 0 m – 46.7 m |
| `middle` | 46.7 m – 93.3 m |
| `attacking` | 93.3 m – 140 m |

Each player's y coordinate (in metres) determines which zone they fall into. For example, a player at y = 60 m is in the `middle` zone.

Referees and ignored tracks are excluded. The output is a count per team per zone, like: `{"ellistown": {"defensive": 2, "middle": 5, "attacking": 4}, "opposition": {...}}`.

---

## `compute_clip_summary(positions, team_assignments, fps) → dict`

**What it does:** Orchestrates everything above — calls each of the three functions and assembles the results into one complete summary dict that the API returns.

**Step by step:**

1. **Key normalisation:** JSON dictionaries always have string keys (like `"42"`), but `track_id` values in the positions list are integers. This step converts all keys in `team_assignments` from strings to integers so lookups work correctly.

2. **Player distances:** Calls `compute_player_distances` once to get distance covered for every track.

3. **Per-player summary:** Builds a `per_player` dict that combines the distance figure with the team label for each track.

4. **Frame loop:** Collects every unique `frame_idx` across all positions, sorts them, then iterates through them in order. For each frame, calls both `compute_team_spatial` and `compute_zone_balance`, storing the results keyed by frame index. This produces two "timeseries" — one frame's worth of spatial data per entry.

5. **Aggregation:** After the loop, computes summary statistics across all frames:
   - `centroid_separation_m` mean, min, and max — so you can see whether the gap between teams grew or shrank during the clip.
   - Per-team averages of `spread_m2`, `centroid_x_m`, and `centroid_y_m` — answering "on average, where was the team positioned and how compact were they?"

6. **Clip metadata:** Records `fps`, the clip duration in seconds (`(last_frame - first_frame) / fps`), and the total frame count.

**Returns:** One large dict with four top-level sections: `per_player`, `spatial_timeseries`, `zone_balance_timeseries`, `spatial_summary`, and `clip_meta`.

---

## API integration

| Endpoint | What it does |
|----------|--------------|
| `POST /videos/{id}/compute-kpis` | Runs `compute_clip_summary` on the stored positions and returns the full KPI dict |

The endpoint accepts an optional `?end_frame=N` query parameter. If supplied, any positions with `frame_idx > N` are filtered out before the computation runs. This lets the user trim the trailing frames (e.g. players jogging back after a score is awarded) without re-running tracking or homography — those trailing frames would distort the distance and zone balance numbers.

---

## Design decisions explained

**Why convex-hull area and not standard deviation?**
Standard deviation measures spread along one axis at a time (x spread or y spread separately). Convex-hull area gives the total 2D footprint of the team, regardless of which direction they are spread. It directly answers "how much of the pitch are they occupying?" in square metres — which is much easier to explain to a coach.

**Why centroid separation and not formation shape?**
Formation shape (e.g. 4-3-3 vs 2-4-5) changes from second to second on short clips, and tracking noise makes the exact shape unreliable. Centroid separation is a single robust number that directly answers "were both teams bunched together or spread apart?" — which is the key coaching question for a scoring play.

**Why absolute zone labels instead of "attack" and "defence"?**
The backend does not know which direction Ellistown is attacking in a given clip. That context comes from the video itself, which the backend cannot interpret. Using absolute labels (y < 46.7 m = `defensive`) keeps the backend simple and correct. The frontend reads the player distribution and flips the labels if needed.

**Why is SciPy optional?**
SciPy is a large library. The `from scipy.spatial import ConvexHull` import is done inside the function body, not at the top of the file. That way, if SciPy is not installed, the rest of `kpi.py` still works — `spread_m2` just returns `0.0` for all teams instead of a real area.
