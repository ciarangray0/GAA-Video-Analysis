"""KPI computation endpoint."""
import asyncio
import logging
from functools import partial
from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from pipeline.persistence import load_team_classifications
from store import store
from routes.deps import get_video_or_404

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/videos/{video_id}/compute-kpis")
async def compute_kpis(video_id: str, end_frame: Optional[int] = Query(None)):
    """Compute locomotor and spatial KPIs for all players in the clip."""
    from pipeline.kpi import compute_clip_summary

    get_video_or_404(video_id)

    positions = store.player_positions_cache.get(video_id)
    if not positions:
        raise HTTPException(
            status_code=404,
            detail="No player positions found. Run map_players and interpolate first.",
        )

    team_classifications = (
        store.team_classifications_cache.get(video_id)
        or load_team_classifications(video_id)
        or {}
    )

    fps = store.videos[video_id].get("fps") or 25.0

    if end_frame is not None:
        positions = [p for p in positions if p.frame_idx <= end_frame]

    pos_dicts = [
        {
            "frame_idx": p.frame_idx,
            "track_id": p.track_id,
            "x_pitch": p.x_pitch,
            "y_pitch": p.y_pitch,
        }
        for p in positions
    ]

    summary = await asyncio.get_event_loop().run_in_executor(
        None,
        partial(compute_clip_summary, pos_dicts, team_classifications, float(fps)),
    )

    return summary