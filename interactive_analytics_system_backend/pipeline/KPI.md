# KPI Module (`kpi.py`)

Computes spatial and locomotor KPIs for a GAA scoring clip. Designed for short clips (10–30 s) where the coaching question is *where were players and how did space open up*, not GPS workload assessment.

Speed-zone / GPS-workload metrics (HSR, sprint distance, accelerations) are intentionally omitted — on clips of this length they are dominated by noise from the homography projection and do not answer spatial questions.

---

## Constants

| Constant | Value | Description |
|----------|-------|-------------|
| `PX_PER_METRE` | `10.0` | Pitch canvas scale (850×1400 px = 85×140 m) |
| `PITCH_LENGTH_M` | `140.0` | Full pitch length in meters |
| `THIRD_1_END_M` | `46.667` | End of the first (defensive) third |
| `THIRD_2_END_M` | `93.333` | End of the second (middle) third |

Pitch thirds are absolute backend labels along the y-axis (y=0 is one endline, y=140 is the other). The frontend maps these zones to attack/defence meaning depending on which end Ellistown is attacking.

---

## Helper Functions

### `_get(p, key)`
Reads `key` from `p` whether `p` is a dict or a Pydantic model. Allows the module to accept either `PlayerPitchPosition` objects or plain dicts.

### `_team_label(cls) → str`
Extracts the `team` string from a classification dict or Pydantic object. Returns `"unclassified"` if `cls` is `None`.

---

## `compute_player_distances(positions) → Dict[int, dict]`

Computes the total distance (in meters) covered by each player track over the full clip.

**Algorithm:**
1. Group positions by `track_id`, keeping `(frame_idx, x_pitch, y_pitch)`.
2. Sort each track's positions by `frame_idx`.
3. Compute consecutive displacements: `dx = diff(x) / PX_PER_METRE`, `dy = diff(y) / PX_PER_METRE`.
4. Sum Euclidean displacement: `total = sum(sqrt(dx² + dy²))`.

Frame gaps wider than 1 (e.g. after interpolation dropped a run) are handled correctly because only the spatial displacement is summed, not speed × time.

**Returns:** `{track_id: {total_distance_m: float}}`.

---

## `compute_team_spatial(positions, team_assignments, frame_idx) → dict`

Computes centroid and convex-hull spread per team for a **single frame**.

**Algorithm:**
1. Filter `positions` to `frame_idx`.
2. Group player positions by team (via `team_assignments`). Tracks labelled `'referee'` or `'ignore'` are excluded.
3. Convert pixel coords to meters: `x_m = x_pitch / 10`, `y_m = y_pitch / 10`.
4. Per team: compute centroid `(mean_x_m, mean_y_m)`.
5. If SciPy is available and the team has ≥ 3 players: compute convex-hull area (`ConvexHull.volume` in 2-D = area) as `spread_m²`. Falls back to `0.0` if hull construction fails (e.g. collinear points).
6. If both `'ellistown'` and `'opposition'` are present: compute 2D centroid separation `sqrt((ex-ox)² + (ey-oy)²)`.

**Returns:**
```python
{
  "teams": {
    "ellistown":  {"centroid_x_m", "centroid_y_m", "spread_m2", "num_players_visible"},
    "opposition": {"centroid_x_m", "centroid_y_m", "spread_m2", "num_players_visible"},
  },
  "centroid_separation_m": float | None
}
```

---

## `compute_zone_balance(positions, team_assignments, frame_idx) → dict`

Counts how many players each team has in each pitch third for a **single frame**.

**Zone boundaries (absolute y-axis):**
| Zone | y range |
|------|---------|
| `defensive` | 0 – 46.7 m |
| `middle` | 46.7 – 93.3 m |
| `attacking` | 93.3 – 140 m |

These are absolute backend labels. The frontend interprets them relative to which end Ellistown is attacking. Referees and ignored tracks are excluded.

**Returns:** `{team: {defensive: int, middle: int, attacking: int}}`.

---

## `compute_clip_summary(positions, team_assignments, fps) → dict`

Orchestrates the full KPI computation over all frames in the clip.

**Steps:**
1. Normalise `team_assignments` keys to `int` (JSON keys come in as strings).
2. Call `compute_player_distances` for locomotor metrics.
3. Build `per_player`: merge distance metrics with team label for each track.
4. Collect all unique `frame_idx` values from `positions` (sorted).
5. For each frame: call `compute_team_spatial` and `compute_zone_balance` to build per-frame timeseries dicts (`spatial_timeseries`, `zone_balance_timeseries`).
6. Aggregate `centroid_separation_m` over all frames: compute `mean`, `min`, `max`.
7. Per-team summary: aggregate `spread_m2`, `centroid_x_m`, `centroid_y_m` means across all frames the team is present.
8. `clip_meta`: `fps`, `duration_s = (last_frame - first_frame) / fps`, `total_frames`.

**Returns:**
```python
{
  "per_player": {
    str(track_id): {"team": str, "total_distance_m": float}
  },
  "spatial_timeseries": {
    str(frame_idx): {
      "teams": {
        team: {"centroid_x_m", "centroid_y_m", "spread_m2", "num_players_visible"}
      },
      "centroid_separation_m": float | None
    }
  },
  "zone_balance_timeseries": {
    str(frame_idx): {team: {"defensive": int, "middle": int, "attacking": int}}
  },
  "spatial_summary": {
    "centroid_separation_m": {"mean": float, "min": float, "max": float},
    "per_team": {team: {"mean_spread_m2", "mean_centroid_x_m", "mean_centroid_y_m"}}
  },
  "clip_meta": {"fps": float, "duration_s": float, "total_frames": int}
}
```

---

## API Integration

| Endpoint | Action |
|----------|--------|
| `POST /videos/{id}/compute-kpis` | Calls `compute_clip_summary`; accepts optional `?end_frame=N` query param to trim the clip |

The `end_frame` parameter allows the frontend to exclude trailing frames (e.g. players jogging back after a score) from KPI computation without re-running tracking or homography. When supplied, positions with `frame_idx > end_frame` are filtered out before `compute_clip_summary` is called.

---

## Design Notes

**Why convex-hull spread and not standard deviation?**
Convex-hull area captures the actual footprint of the team on the pitch — it answers "how much space did they cover?" Standard deviation of coordinates is sensitive to which direction the spread occurs and is harder to interpret physically.

**Why only centroid separation, not formation shape?**
For clips of 10–30 s, formation shape changes too rapidly and tracking noise obscures fine-grained shape metrics. Centroid separation is robust and directly answers the coaching question: "how compressed was the contest?"

**Why absolute zone labels (defensive/middle/attacking) rather than relative?**
The backend has no knowledge of which end Ellistown is attacking in a given clip — that context is inferred by the frontend from the distribution of player positions. Absolute labels are stable and consistent. The frontend `detectedZone` and `clipMode` logic handles the interpretation.

**SciPy optional dependency:**
`from scipy.spatial import ConvexHull` is imported inside `compute_team_spatial` to avoid a hard dependency at module load time. If SciPy is not installed, `spread_m2` defaults to `0.0` for all teams.
