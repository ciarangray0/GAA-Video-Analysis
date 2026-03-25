"""KPI computation for GAA video clips.

Designed for short scoring clips (10-30 s) where the coaching question is
*where were players and how did space open up*, not workload assessment.

Speed-zone / GPS-workload metrics (HSR, sprint distance, accelerations) are
intentionally omitted: on clips of this length they are dominated by noise
from the homography projection and don't answer spatial questions.

What is computed:
  - Total distance per player (simple displacement sum, already robust).
  - Team spread (convex-hull area) per frame — how compact or stretched each
    team was at every moment.
  - Centroid separation per frame — distance between the two team centroids,
    showing compression/expansion as the score developed.
  - Zone balance per frame — how many players each team had in each pitch
    third at every moment.
  - Summary statistics (mean/min/max) aggregated over the clip.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PX_PER_METRE = 10.0           # 10 px == 1 m  (850×1400 canvas == 85×140 m)

PITCH_LENGTH_M = 140.0
THIRD_1_END_M = PITCH_LENGTH_M / 3.0        # 46.667 m — end of defensive third
THIRD_2_END_M = 2.0 * PITCH_LENGTH_M / 3.0  # 93.333 m — end of middle third


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get(p, key: str):
    return p[key] if isinstance(p, dict) else getattr(p, key)


def _team_label(cls) -> str:
    if cls is None:
        return "unclassified"
    return cls["team"] if isinstance(cls, dict) else getattr(cls, "team", "unclassified")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_player_distances(positions: List) -> Dict[int, dict]:
    """Total distance covered (m) per track over the full clip.

    Uses straight-line displacement between consecutive observed positions.
    Frame gaps wider than 1 (e.g. after interpolation dropped a run) are
    handled correctly because we only sum the spatial displacement, not
    speed × time.
    """
    tracks: Dict[int, List[tuple]] = {}
    for p in positions:
        tid = int(_get(p, "track_id"))
        fidx = int(_get(p, "frame_idx"))
        xp = float(_get(p, "x_pitch"))
        yp = float(_get(p, "y_pitch"))
        tracks.setdefault(tid, []).append((fidx, xp, yp))

    result: Dict[int, dict] = {}
    for tid, pts in tracks.items():
        pts.sort(key=lambda t: t[0])
        xs = np.array([t[1] for t in pts], dtype=float)
        ys = np.array([t[2] for t in pts], dtype=float)
        if len(pts) < 2:
            result[tid] = {"total_distance_m": 0.0}
            continue
        dx = np.diff(xs) / PX_PER_METRE
        dy = np.diff(ys) / PX_PER_METRE
        result[tid] = {"total_distance_m": round(float(np.sqrt(dx**2 + dy**2).sum()), 2)}

    return result


def compute_team_spatial(
    positions: List,
    team_assignments: Dict,
    frame_idx: int,
) -> Dict[str, dict]:
    """Centroid and convex-hull spread per team for a single frame.

    Returns
    -------
    Dict of team_name → {centroid_x_m, centroid_y_m, spread_m2,
                          num_players_visible}
    Also includes a top-level ``centroid_separation_m`` key when both
    'ellistown' and 'opposition' are present.
    """
    try:
        from scipy.spatial import ConvexHull  # type: ignore
        _has_scipy = True
    except ImportError:
        _has_scipy = False

    frame_pos = [p for p in positions if int(_get(p, "frame_idx")) == frame_idx]

    teams: Dict[str, List[tuple]] = {}
    for p in frame_pos:
        tid = int(_get(p, "track_id"))
        xp = float(_get(p, "x_pitch"))
        yp = float(_get(p, "y_pitch"))
        cls = team_assignments.get(tid) or team_assignments.get(str(tid))
        team = _team_label(cls)
        if team in ("referee", "ignore"):
            continue
        teams.setdefault(team, []).append((xp / PX_PER_METRE, yp / PX_PER_METRE))

    per_team: Dict[str, dict] = {}
    for team, pts in teams.items():
        arr = np.array(pts)
        cx = float(arr[:, 0].mean())
        cy = float(arr[:, 1].mean())
        spread = 0.0
        if _has_scipy and len(pts) >= 3:
            try:
                hull = ConvexHull(arr)
                spread = float(hull.volume)  # ConvexHull.volume == area in 2-D
            except Exception:
                spread = 0.0
        per_team[team] = {
            "centroid_x_m": round(cx, 2),
            "centroid_y_m": round(cy, 2),
            "spread_m2": round(spread, 2),
            "num_players_visible": len(pts),
        }

    result: dict = {"teams": per_team}

    # Inter-team centroid separation
    if "ellistown" in per_team and "opposition" in per_team:
        e = per_team["ellistown"]
        o = per_team["opposition"]
        sep = float(np.hypot(e["centroid_x_m"] - o["centroid_x_m"],
                             e["centroid_y_m"] - o["centroid_y_m"]))
        result["centroid_separation_m"] = round(sep, 2)
    else:
        result["centroid_separation_m"] = None

    return result


def compute_zone_balance(
    positions: List,
    team_assignments: Dict,
    frame_idx: int,
) -> Dict[str, dict]:
    """Player counts per team per pitch third for a single frame.

    Thirds along the y-axis (pitch length):
      defensive:  0 – 46.7 m
      middle:    46.7 – 93.3 m
      attacking: 93.3 – 140 m
    """
    frame_pos = [p for p in positions if int(_get(p, "frame_idx")) == frame_idx]
    teams: Dict[str, Dict[str, int]] = {}

    for p in frame_pos:
        tid = int(_get(p, "track_id"))
        yp = float(_get(p, "y_pitch"))
        cls = team_assignments.get(tid) or team_assignments.get(str(tid))
        team = _team_label(cls)
        if team in ("referee", "ignore"):
            continue

        y_m = yp / PX_PER_METRE
        zone = "defensive" if y_m < THIRD_1_END_M else ("middle" if y_m < THIRD_2_END_M else "attacking")

        if team not in teams:
            teams[team] = {"defensive": 0, "middle": 0, "attacking": 0}
        teams[team][zone] += 1

    return teams


def compute_clip_summary(
    positions: List,
    team_assignments: Dict,
    fps: float,
) -> dict:
    """Full spatial KPI summary for a clip.

    Returns
    -------
    {
      per_player:              {track_id: {team, total_distance_m}},
      spatial_timeseries:      {frame_idx: {teams: {...}, centroid_separation_m}},
      zone_balance_timeseries: {frame_idx: {team: {defensive, middle, attacking}}},
      spatial_summary:         {centroid_separation_m: {mean, min, max},
                                per_team: {team: {mean_spread_m2,
                                                   mean_centroid_x_m,
                                                   mean_centroid_y_m}}},
      clip_meta:               {fps, duration_s, total_frames},
    }
    """
    # Normalise team_assignments keys to int
    ta: Dict[int, dict] = {}
    for k, v in team_assignments.items():
        try:
            ta[int(k)] = v
        except (TypeError, ValueError):
            pass

    distances = compute_player_distances(positions)

    per_player: Dict[str, dict] = {}
    for tid, metrics in distances.items():
        cls = ta.get(tid)
        per_player[str(tid)] = {"team": _team_label(cls), **metrics}

    frame_indices = sorted(set(int(_get(p, "frame_idx")) for p in positions))

    spatial_ts: Dict[str, dict] = {}
    zone_ts: Dict[str, dict] = {}
    for fidx in frame_indices:
        spatial_ts[str(fidx)] = compute_team_spatial(positions, ta, fidx)
        zone_ts[str(fidx)] = compute_zone_balance(positions, ta, fidx)

    # Aggregate summary statistics
    separations = [
        v["centroid_separation_m"]
        for v in spatial_ts.values()
        if v["centroid_separation_m"] is not None
    ]
    sep_summary: dict = {}
    if separations:
        sep_summary = {
            "mean": round(float(np.mean(separations)), 2),
            "min": round(float(np.min(separations)), 2),
            "max": round(float(np.max(separations)), 2),
        }

    team_names = set()
    for frame_data in spatial_ts.values():
        team_names.update(frame_data["teams"].keys())

    per_team_summary: Dict[str, dict] = {}
    for team in team_names:
        spreads, cxs, cys = [], [], []
        for frame_data in spatial_ts.values():
            td = frame_data["teams"].get(team)
            if td:
                spreads.append(td["spread_m2"])
                cxs.append(td["centroid_x_m"])
                cys.append(td["centroid_y_m"])
        if spreads:
            per_team_summary[team] = {
                "mean_spread_m2": round(float(np.mean(spreads)), 2),
                "mean_centroid_x_m": round(float(np.mean(cxs)), 2),
                "mean_centroid_y_m": round(float(np.mean(cys)), 2),
            }

    total_frames = len(frame_indices)
    duration_s = (frame_indices[-1] - frame_indices[0]) / fps if fps > 0 and len(frame_indices) > 1 else 0.0

    return {
        "per_player": per_player,
        "spatial_timeseries": spatial_ts,
        "zone_balance_timeseries": zone_ts,
        "spatial_summary": {
            "centroid_separation_m": sep_summary,
            "per_team": per_team_summary,
        },
        "clip_meta": {
            "fps": fps,
            "duration_s": round(duration_s, 2),
            "total_frames": total_frames,
        },
    }
