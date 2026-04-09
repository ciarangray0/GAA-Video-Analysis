"""Team classification endpoints."""
import asyncio
import logging

import numpy as np
from fastapi import APIRouter, HTTPException

from pipeline.schemas import TeamOverrideRequest, VALID_TEAMS
from pipeline.map_players import filter_detections_for_mapping
from pipeline.persistence import load_detections, load_team_classifications, save_team_classifications
from pipeline.team_classifier import override_classification
from store import store
from routes.deps import get_video_or_404

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/videos/{video_id}/classify-teams")
async def classify_teams(video_id: str):
    """Classify player tracks into teams by jersey colour analysis.

    Samples video frames for each track, extracts the jersey HSV colour,
    and assigns 'ellistown' (yellow jersey) or 'opposition'.
    Returns per-track classifications plus a summary with cluster statistics.
    """
    # NOTE: classify_tracks is kept as a lazy import — it pulls in heavy colour analysis deps.
    from pipeline.team_classifier import classify_tracks

    video_info = get_video_or_404(video_id)

    detections = store.detections_cache.get(video_id) or load_detections(video_id)
    if detections is None:
        raise HTTPException(status_code=400, detail="No detections found. Run tracking first.")

    player_detections = filter_detections_for_mapping(detections)

    try:
        classifications = await asyncio.to_thread(
            classify_tracks, video_info["path"], player_detections
        )
    except Exception as e:
        logger.error(f"Team classification failed for video {video_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Team classification failed: {str(e)}")

    store.team_classifications_cache[video_id] = classifications
    save_team_classifications(video_id, classifications)

    confidences = [v["confidence"] for v in classifications.values()]
    ellistown_ids = [tid for tid, v in classifications.items() if v["team"] == "ellistown"]
    opposition_ids = [tid for tid, v in classifications.items() if v["team"] == "opposition"]
    low_conf_ids = [tid for tid, v in classifications.items() if v["confidence"] < 0.6]

    hsv_separation = None
    if ellistown_ids and opposition_ids:
        ell_mean = np.mean([classifications[tid]["mean_hsv"] for tid in ellistown_ids], axis=0)
        opp_mean = np.mean([classifications[tid]["mean_hsv"] for tid in opposition_ids], axis=0)
        hsv_separation = round(float(np.linalg.norm(ell_mean - opp_mean)), 2)

    logger.info(
        f"Team classification for {video_id}: {len(ellistown_ids)} ellistown, "
        f"{len(opposition_ids)} opposition, separation={hsv_separation}"
    )

    return {
        "classifications": {str(k): v for k, v in classifications.items()},
        "summary": {
            "num_ellistown": len(ellistown_ids),
            "num_opposition": len(opposition_ids),
            "num_referee": 0,
            "mean_confidence": round(float(np.mean(confidences)) if confidences else 0.0, 3),
            "low_confidence_tracks": low_conf_ids,
            "hsv_cluster_separation": hsv_separation,
        },
    }


@router.get("/videos/{video_id}/classify-teams")
async def get_team_classifications(video_id: str):
    """Return stored team classifications for a video."""
    get_video_or_404(video_id)

    classifications = (
        store.team_classifications_cache.get(video_id)
        or load_team_classifications(video_id)
    )
    if classifications is None:
        raise HTTPException(status_code=404, detail="No team classifications found. Run classify-teams first.")

    return {"classifications": {str(k): v for k, v in classifications.items()}}


@router.patch("/videos/{video_id}/classify-teams")
async def override_team_classification(video_id: str, body: TeamOverrideRequest):
    """Override a single track's team assignment."""
    get_video_or_404(video_id)

    if body.team not in VALID_TEAMS:
        raise HTTPException(
            status_code=400,
            detail=f"team must be one of: {', '.join(sorted(VALID_TEAMS))}",
        )

    classifications = (
        store.team_classifications_cache.get(video_id)
        or load_team_classifications(video_id)
        or {}
    )

    classifications = override_classification(classifications, body.track_id, body.team)
    store.team_classifications_cache[video_id] = classifications
    save_team_classifications(video_id, classifications)

    logger.info(f"Track {body.track_id} reassigned to '{body.team}' for video {video_id}")
    return {"classifications": {str(k): v for k, v in classifications.items()}}