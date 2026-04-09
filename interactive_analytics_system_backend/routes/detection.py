"""Detection and tracking endpoints."""
import logging
from typing import List

import cv2
import numpy as np
from fastapi import APIRouter, HTTPException
from fastapi.responses import Response

from pipeline.schemas import Detection, TrackResponse
from pipeline.persistence import load_detections, save_detections
from store import store
from routes.deps import get_video_or_404

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/videos/{video_id}/detections", response_model=List[Detection])
async def get_video_detections(video_id: str):
    """Return the raw YOLO+BotSort detections for a video."""
    get_video_or_404(video_id)

    detections = store.detections_cache.get(video_id) or load_detections(video_id)
    if detections is None:
        raise HTTPException(status_code=404, detail="No detections found. Run tracking first.")
    return detections


@router.get("/videos/{video_id}/frames/{frame_idx}/detections_overlay")
async def get_detections_overlay(video_id: str, frame_idx: int):
    """Return a JPEG of the raw video frame with BotSort bounding boxes overlaid."""
    video_info = get_video_or_404(video_id)

    detections = store.detections_cache.get(video_id) or load_detections(video_id)
    if detections is None:
        raise HTTPException(status_code=404, detail="No detections found. Run tracking first.")

    try:
        fps = video_info["fps"] or 25
        cap = cv2.VideoCapture(video_info["path"])
        cap.set(cv2.CAP_PROP_POS_MSEC, frame_idx / fps * 1000)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            raise HTTPException(status_code=500, detail="Failed to extract frame")

        frame_dets = [d for d in detections if d.frame_idx == frame_idx]
        for det in frame_dets:
            x1, y1, x2, y2 = int(det.x1), int(det.y1), int(det.x2), int(det.y2)
            hue = int((det.track_id * 137.508) % 180)
            colour_hsv = np.array([[[hue, 220, 220]]], dtype=np.uint8)
            colour_bgr = cv2.cvtColor(colour_hsv, cv2.COLOR_HSV2BGR)[0][0].tolist()
            cv2.rectangle(frame, (x1, y1), (x2, y2), colour_bgr, 2)
            label = f"#{det.track_id}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), colour_bgr, -1)
            cv2.putText(frame, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return Response(content=buffer.tobytes(), media_type="image/jpeg", headers={"Cache-Control": "max-age=3600"})
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed detections_overlay {frame_idx} for video {video_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create overlay: {str(e)}")


@router.post("/videos/{video_id}/track", response_model=TrackResponse)
async def track_video(video_id: str):
    """Run YOLO + BotSort tracking on the video."""
    video_info = get_video_or_404(video_id)
    video_path = video_info["path"]

    detections = load_detections(video_id)
    if detections is None:
        try:
            # Lazy import: gpu_inference pulls in heavy ML deps; only loaded when tracking is triggered.
            from gpu_inference import get_gpu_client
            client = get_gpu_client()
            logger.info(f"Running tracking on remote GPU ({client.provider.value}) for video {video_id}")
            detections = client.track_video(video_path)
            store.detections_cache[video_id] = detections
            save_detections(video_id, detections)
        except Exception as e:
            logger.error(f"Tracking failed for video {video_id}: {e}")
            raise HTTPException(status_code=500, detail=f"Tracking failed: {str(e)}")
    else:
        store.detections_cache[video_id] = detections

    if not detections:
        raise HTTPException(status_code=500, detail="No detections found in video")

    unique_tracks = len(set(d.track_id for d in detections))
    frames_processed = max(d.frame_idx for d in detections) + 1
    logger.info(f"Tracking complete for {video_id}: {frames_processed} frames, {unique_tracks} tracks")

    return TrackResponse(frames_processed=frames_processed, tracks=unique_tracks)