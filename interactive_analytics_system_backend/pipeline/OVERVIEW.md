# Pipeline Module Overview

The `pipeline/` package contains all the data-processing logic. It is deliberately free of HTTP/FastAPI concerns — modules only deal with numpy arrays, Python data structures, and file paths.

---

## Module Dependency Graph

```
app.py  (creates FastAPI app, registers routers from routes/)
 │
 routes/
 ├── routes.videos        → pipeline.video, pipeline.rendering, pipeline.persistence
 ├── routes.detection     → gpu_inference (lazy import), pipeline.persistence
 │                              gpu_inference  (GPUInferenceClient for Modal)
 ├── routes.homography    → pipeline.homography, pipeline.constrained_homography,
 │                              pipeline.persistence
 ├── routes.mapping       → pipeline.map_players, pipeline.trajectories,
 │                              pipeline.persistence
 ├── routes.classification → pipeline.team_classifier, pipeline.persistence
 └── routes.kpi           → pipeline.kpi, pipeline.persistence
 │
 pipeline/
 ├── pipeline.config          (OUT_W, OUT_H, YOLO_MODEL_PATH, DEFAULT_CONF)
 ├── pipeline.gaa_pitch_config (pitch geometry: vertices, lines, sidelines)
 ├── pipeline.schemas         (Pydantic models for all pipeline types)
 ├── pipeline.persistence     (all disk I/O: save/load JSON, homographies, annotations)
 ├── pipeline.video           (get_video_metadata, extract_frame)
 ├── pipeline.rendering       (warp_frame)
 ├── pipeline.homography      (compute_homographies_with_lines_v3, resolve_pitch_coordinates)
 │    ├── pipeline.config
 │    ├── pipeline.gaa_pitch_config
 │    ├── pipeline.schemas
 │    └── pipeline.line_constraints (sample_points_on_line, GAA_PITCH_LINES/SIDELINES)
 ├── pipeline.constrained_homography (build_optical_flow_per_frame_H)
 ├── pipeline.map_players     (filter_detections_for_mapping, map_players_to_pitch)
 │    └── pipeline.homography (map_pixel_to_pitch)
 ├── pipeline.trajectories    (interpolate_trajectories)
 │    └── pipeline.config      (OUT_W, OUT_H for canvas clipping)
 ├── pipeline.team_classifier (classify_tracks, override_classification)
 │    └── pipeline.schemas     (Detection)
 └── pipeline.kpi             (compute_clip_summary, compute_player_distances,
                                compute_team_spatial, compute_zone_balance)
```

---

## Module Responsibilities

| Module | What it does |
|--------|-------------|
| `config.py` | Canvas size constants, model path, tracking confidence |
| `gaa_pitch_config.py` | All pitch geometry: vertices, horizontal lines, vertical sidelines |
| `schemas.py` | Pydantic models: annotations in, positions + detections out; team override request |
| `persistence.py` | All disk I/O: save/load detections, homographies, annotations, team classifications |
| `video.py` | OpenCV wrappers: metadata extraction, single-frame extraction |
| `rendering.py` | `warp_frame` — `cv2.warpPerspective` wrapper |
| `homography.py` | Anchor H computation (RANSAC + weighted DLT), `resolve_pitch_coordinates`, `map_pixel_to_pitch` |
| `line_constraints.py` | `sample_points_on_line`, re-exports `GAA_PITCH_LINES`, `GAA_PITCH_SIDELINES` |
| `constrained_homography.py` | `build_optical_flow_per_frame_H` — LK flow, drift correction, SG smoothing |
| `map_players.py` | Filter ball/referee, map bottom-centre of each bbox through per-frame H |
| `trajectories.py` | Linear interp → Savitzky-Golay → max-velocity clamp → canvas clip |
| `team_classifier.py` | Jersey-colour HSV analysis; classifies each track as Ellistown or opposition |
| `kpi.py` | Spatial + locomotor KPI computation: distances, centroid separation, convex-hull spread, zone balance |

---

## Shared Conventions

### Coordinate system
All modules that deal with positions operate in **pitch-canvas pixel space** (0..850 × 0..1400). Meter values are only ever used when setting up homography destination points (in `homography.py`) — they are converted to canvas pixels immediately and never propagated further.

### Homography matrices
All H matrices are `(3, 3) float64` numpy arrays mapping **image pixels → pitch-canvas pixels** via `H @ [x, y, 1]ᵀ` followed by perspective division by the third coordinate.

### Frame indexing
All frame indices are 0-based integers. Dictionaries keyed by frame index use `int` keys internally; they are serialised as strings in JSON (`str(k)`) because JSON requires string keys.

### Scale constant
`OUT_W / GAA_PITCH_WIDTH = OUT_H / GAA_PITCH_LENGTH = 10 px/m` exactly. This is relied upon throughout but never stored as a named constant — use the division formula consistently.

---

## How Modules Chain Together (Processing Order)

```
1. video.py         extract_frame / get_video_metadata
2. gpu_inference/   get_gpu_client().track_video()  → List[Detection]
   (dispatched from routes/detection.py — no detect.py in pipeline/ any more)
3. homography.py    compute_homographies_with_lines_v3
                   → Dict[frame_idx, H]  (anchor frames only)
4. constrained_homography.py
                   build_optical_flow_per_frame_H
                   → Dict[frame_idx, H]  (every frame)
5. map_players.py  filter_detections_for_mapping
                   map_players_to_pitch
                   → List[PlayerPitchPosition]  (sparse, one per detection)
6. trajectories.py interpolate_trajectories
                   → List[PlayerPitchPosition]  (dense, every frame in range)
7. team_classifier.py  classify_tracks            (optional, post-processing)
                       → Dict[track_id, {team, confidence, mean_hsv}]

8. kpi.py              compute_clip_summary        (optional, post-processing)
                       → {per_player, spatial_timeseries,
                          zone_balance_timeseries, spatial_summary, clip_meta}
```

Steps 3 and 4 are triggered together by the `POST /homographies/v3` endpoint. Steps 5 and 6 are separate endpoints called in order by the frontend. Step 7 is an independent optional step triggered by the "Classify Teams" button in the frontend — it does not depend on the trajectory data and can be run at any point after tracking. Step 8 is triggered by "Compute KPIs" in the frontend after team classification; it accepts an optional `end_frame` parameter to exclude trailing frames from analysis.
