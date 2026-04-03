"""Player mapping and trajectory interpolation endpoints."""
import logging
from typing import List

from fastapi import APIRouter, HTTPException, Query

from pipeline.schemas import PlayerPitchPosition, InterpolationResponse
from pipeline.map_players import map_players_to_pitch, filter_detections_for_mapping
from pipeline.trajectories import interpolate_trajectories
from pipeline.persistence import load_detections, load_homography_dict
from store import store
from routes.deps import get_video_or_404

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/videos/{video_id}/map_players", response_model=List[PlayerPitchPosition])
async def map_players(video_id: str):
    """Map player detections to pitch coordinates using computed homographies."""
    get_video_or_404(video_id)

    detections = load_detections(video_id)
    if detections is None:
        raise HTTPException(status_code=400, detail="No detections found. Run tracking first.")

    homographies = store.v3_per_frame_H_cache.get(video_id) or load_homography_dict(video_id, "v3_homographies")
    if homographies is None:
        raise HTTPException(status_code=400, detail="No homographies found. Compute homographies first.")

    anchor_hs = store.v3_anchor_H_cache.get(video_id) or load_homography_dict(video_id, "v3_anchor_homographies")
    anchor_frame_indices = set(anchor_hs.keys()) if anchor_hs else None

    try:
        detections = filter_detections_for_mapping(detections)
        positions = map_players_to_pitch(
            detections, homographies,
            anchor_frame_indices=anchor_frame_indices,
        )
        store.player_positions_cache[video_id] = positions
    except Exception as e:
        logger.error(f"Player mapping failed for video {video_id}: {e}")
        raise HTTPException(status_code=500, detail="Player mapping failed")

    logger.info(f"Mapped {len(positions)} player positions for video {video_id}")
    return positions


@router.post("/videos/{video_id}/interpolate", response_model=InterpolationResponse)
async def interpolate_trajectories_endpoint(
    video_id: str,
    start_frame: int = Query(0, description="First frame to interpolate"),
    end_frame: int = Query(100, description="Last frame to interpolate (inclusive)"),
    sg_long_window: int = Query(15, ge=3, le=51, description="SG window for tracks >20 frames"),
    sg_mid_window: int = Query(11, ge=3, le=31, description="SG window for tracks 10-20 frames"),
    max_vel_px: float = Query(4.0, ge=0.0, le=50.0, description="Max displacement per frame in pitch-canvas pixels (0 = disabled)"),
):
    """Interpolate player trajectories between anchor frames.

    Smoothing pipeline per track:
    1. Linear interpolation fills gaps between detections.
    2. Savitzky-Golay smoothing (window size depends on track length).
    3. Max-velocity clamping removes residual spikes.
    """
    get_video_or_404(video_id)

    if start_frame < 0 or end_frame < start_frame:
        raise HTTPException(status_code=400, detail="Invalid frame range")

    sparse_positions = store.player_positions_cache.get(video_id)
    if sparse_positions is None:
        raise HTTPException(status_code=400, detail="No player positions found. Run map_players first.")

    homography_positions = [p for p in sparse_positions if p.source in ("homography", "homography_interp")]
    if not homography_positions:
        raise HTTPException(status_code=400, detail="No homography-based positions found for interpolation")

    try:
        interpolated = interpolate_trajectories(
            homography_positions, start_frame, end_frame,
            sg_long_window=sg_long_window,
            sg_mid_window=sg_mid_window,
            max_vel_px=max_vel_px,
        )

        existing_filtered = [
            p for p in sparse_positions
            if not (start_frame <= p.frame_idx <= end_frame)
        ]
        store.player_positions_cache[video_id] = existing_filtered + interpolated

        frames_generated = sum(1 for p in interpolated if p.source == "interpolated")
    except Exception as e:
        logger.error(f"Interpolation failed for video {video_id}: {e}")
        raise HTTPException(status_code=500, detail="Interpolation failed")

    logger.info(f"Generated {frames_generated} interpolated frames for video {video_id}")
    return InterpolationResponse(frames_generated=frames_generated, method="linear")


@router.get("/videos/{video_id}/players", response_model=List[PlayerPitchPosition])
async def get_player_positions(video_id: str):
    """Get all player positions (sparse + interpolated) for a video."""
    get_video_or_404(video_id)

    positions = store.player_positions_cache.get(video_id)
    if positions is None:
        raise HTTPException(status_code=404, detail="No player positions found. Run map_players first.")

    return sorted(positions, key=lambda p: (p.frame_idx, p.track_id))