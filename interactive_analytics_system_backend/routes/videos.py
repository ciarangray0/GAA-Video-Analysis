"""Video upload and frame serving endpoints."""
import logging
import os
import uuid

import cv2
from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from fastapi.responses import Response

from pipeline.config import OUT_W, OUT_H
from pipeline.rendering import draw_reference_lines
from pipeline.schemas import VideoCreateResponse
from pipeline.video import get_video_metadata, extract_frame
from pipeline.persistence import save_video_meta, save_video_file, load_homography_dict, load_team_classifications
from store import store
from routes.deps import get_video_or_404

logger = logging.getLogger(__name__)

router = APIRouter()

MAX_VIDEO_SIZE_MB = int(os.getenv("MAX_VIDEO_SIZE_MB", "500"))
MAX_VIDEO_SIZE = MAX_VIDEO_SIZE_MB * 1024 * 1024


def validate_video_upload(file: UploadFile, content: bytes) -> None:
    """Raise HTTPException if the upload exceeds size limit or is not an MP4."""
    if len(content) > MAX_VIDEO_SIZE:
        raise HTTPException(status_code=413, detail=f"File too large. Maximum size is {MAX_VIDEO_SIZE_MB}MB")
    if not file.filename or not file.filename.lower().endswith(".mp4"):
        raise HTTPException(status_code=400, detail="Only MP4 video files are accepted")
    if file.content_type and file.content_type not in ["video/mp4", "application/octet-stream"]:
        raise HTTPException(status_code=400, detail=f"Invalid content type: {file.content_type}. Expected video/mp4")


def resolve_homography(video_id: str, frame_idx: int):
    """Return the per-frame v3 homography for a frame (nearest if exact frame missing)."""
    v3_hs = store.v3_per_frame_H_cache.get(video_id) or load_homography_dict(video_id, "v3_homographies")
    if not v3_hs:
        return None
    if frame_idx in v3_hs:
        return v3_hs[frame_idx]
    return v3_hs[min(v3_hs.keys(), key=lambda f: abs(f - frame_idx))]


@router.post("/videos", response_model=VideoCreateResponse)
async def upload_video(file: UploadFile = File(...)):
    """Upload a video file and extract metadata."""
    content = await file.read()
    validate_video_upload(file, content)

    video_id = str(uuid.uuid4())
    video_path = save_video_file(video_id, content)

    try:
        metadata = get_video_metadata(str(video_path))
    except Exception as e:
        video_path.unlink()
        logger.error(f"Failed to process video {video_id}: {e}")
        raise HTTPException(status_code=400, detail="Failed to process video. Ensure it is a valid MP4 file.")

    video_meta = {
        "path": str(video_path),
        "fps": metadata["fps"],
        "num_frames": metadata["num_frames"],
        "width": metadata["width"],
        "height": metadata["height"],
        "duration_seconds": metadata["duration_seconds"],
    }
    store.videos[video_id] = video_meta
    save_video_meta(video_id, video_meta)
    logger.info(f"Uploaded video {video_id}: {metadata['num_frames']} frames at {metadata['fps']} fps")

    return VideoCreateResponse(
        video_id=video_id,
        fps=metadata["fps"],
        num_frames=metadata["num_frames"],
        width=metadata["width"],
        height=metadata["height"],
        duration_seconds=metadata["duration_seconds"],
    )


@router.get("/videos/{video_id}/frame/{frame_idx}")
async def get_frame(video_id: str, frame_idx: int):
    """Extract and return a single video frame as JPEG."""
    video_info = get_video_or_404(video_id)

    if frame_idx < 0 or frame_idx >= video_info["num_frames"]:
        raise HTTPException(
            status_code=400,
            detail=f"Frame index must be between 0 and {video_info['num_frames'] - 1}",
        )

    try:
        frame_bytes = extract_frame(video_info["path"], frame_idx)
        if frame_bytes is None:
            raise HTTPException(status_code=500, detail="Failed to extract frame")
        return Response(content=frame_bytes, media_type="image/jpeg", headers={"Cache-Control": "max-age=3600"})
    except Exception as e:
        logger.error(f"Failed to extract frame {frame_idx} from video {video_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to extract frame")


@router.get("/videos/{video_id}/frames/{frame_idx}/warped")
async def get_warped_frame_any(
    video_id: str,
    frame_idx: int,
    players: bool = Query(False, description="Overlay player positions on the warped frame"),
):
    """Return a warped JPEG with pitch reference lines, and optionally player dots.

    Always uses the best available homography (v3 per-frame if computed, else v2
    nearest anchor). Add ``?players=true`` to overlay player positions.
    """
    video_info = get_video_or_404(video_id)

    H = resolve_homography(video_id, frame_idx)
    if H is None:
        raise HTTPException(status_code=400, detail="No homographies computed for this video")

    try:
        cap = cv2.VideoCapture(video_info["path"])
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            raise HTTPException(status_code=500, detail="Failed to extract frame")

        warped = cv2.warpPerspective(frame, H, (OUT_W, OUT_H))
        draw_reference_lines(warped)

        if players:
            classifications = (
                store.team_classifications_cache.get(video_id)
                or load_team_classifications(video_id)
                or {}
            )
            for pos in (p for p in store.player_positions_cache.get(video_id, []) if p.frame_idx == frame_idx):
                team = classifications.get(pos.track_id, {}).get("team", "")
                if team in ("referee", "ignore"):
                    continue
                x, y = int(pos.x_pitch), int(pos.y_pitch)
                if 0 <= x < OUT_W and 0 <= y < OUT_H:
                    if team == "ellistown":
                        color = (0, 210, 255)
                    elif team == "opposition":
                        color = (220, 80, 50)
                    else:
                        color = (0, 0, 255)
                    cv2.circle(warped, (x, y), 8, color, -1)
                    cv2.putText(warped, str(pos.track_id), (x + 10, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        _, buffer = cv2.imencode('.jpg', warped, [cv2.IMWRITE_JPEG_QUALITY, 85])
        cache = "no-cache" if players else "max-age=300"
        return Response(content=buffer.tobytes(), media_type="image/jpeg", headers={"Cache-Control": cache})
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed warped frame {frame_idx} for video {video_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create warped frame: {str(e)}")