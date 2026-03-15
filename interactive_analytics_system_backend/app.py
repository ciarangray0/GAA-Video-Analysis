"""FastAPI application for video analysis pipeline."""
import asyncio
import os
import uuid
from contextlib import asynccontextmanager
from functools import partial
from pathlib import Path
from typing import List, Dict, Optional, Any
import json
import cv2
import numpy as np
import logging

from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

from pipeline.config import OUT_W, OUT_H
from pipeline.rendering import warp_frame
from pipeline.schemas import (
    VideoCreateResponse,
    TrackResponse,
    InterpolationResponse,
    Detection,
    PlayerPitchPosition,
    AnchorFrameAnnotation,
)
# NOTE: `run_tracking` performs heavy ML imports; imported lazily inside endpoint.
from pipeline.homography import compute_homographies_with_lines, resolve_pitch_coordinates
from pipeline.constrained_homography import (
    build_constrained_per_frame_H,
    build_optical_flow_per_frame_H,
)
from pipeline.map_players import map_players_to_pitch
from pipeline.trajectories import interpolate_trajectories
from pipeline.video import get_video_metadata, extract_frame
from pipeline.gaa_pitch_config import GAA_PITCH_WIDTH, GAA_PITCH_LENGTH
from store import store

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MAX_VIDEO_SIZE_MB = int(os.getenv("MAX_VIDEO_SIZE_MB", "500"))
MAX_VIDEO_SIZE = MAX_VIDEO_SIZE_MB * 1024 * 1024
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")
DATA_DIR = Path(os.getenv("DATA_DIR", "data"))

VIDEOS_DIR = DATA_DIR / "videos"
TRACKS_DIR = DATA_DIR / "tracks"
ANNOTATIONS_DIR = DATA_DIR / "annotations"


def _restore_videos_from_disk() -> None:
    """Repopulate store.videos from saved metadata files after a backend restart."""
    for meta_path in VIDEOS_DIR.glob("*_meta.json"):
        meta = _load_json(meta_path)
        if meta is None:
            continue
        video_id = meta.get("video_id")
        video_path = Path(meta.get("path", ""))
        if not video_id or not video_path.exists():
            continue
        store.videos[video_id] = {k: v for k, v in meta.items() if k != "video_id"}
    if store.videos:
        logger.info(f"Restored {len(store.videos)} video(s) from disk on startup")


@asynccontextmanager
async def lifespan(app: FastAPI):
    VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
    TRACKS_DIR.mkdir(parents=True, exist_ok=True)
    ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)
    _restore_videos_from_disk()
    yield


app = FastAPI(title="GAA Video Analysis API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Health Check ---
@app.get("/health")
async def health_check():
    return {"status": "ok"}


# --- Helpers ---

def _get_video_or_404(video_id: str) -> dict:
    if video_id not in store.videos:
        raise HTTPException(status_code=404, detail="Video not found")
    return store.videos[video_id]


def _save_json(path: Path, data) -> None:
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def _load_json(path: Path):
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def _serialize_H(h_dict: Dict[int, np.ndarray]) -> dict:
    return {str(k): v.tolist() for k, v in h_dict.items()}


def _deserialize_H(data: dict) -> Dict[int, np.ndarray]:
    return {int(k): np.array(v) for k, v in data.items()}


def validate_video_upload(file: UploadFile, content: bytes) -> None:
    if len(content) > MAX_VIDEO_SIZE:
        raise HTTPException(status_code=413, detail=f"File too large. Maximum size is {MAX_VIDEO_SIZE_MB}MB")
    if not file.filename or not file.filename.lower().endswith(".mp4"):
        raise HTTPException(status_code=400, detail="Only MP4 video files are accepted")
    if file.content_type and file.content_type not in ["video/mp4", "application/octet-stream"]:
        raise HTTPException(status_code=400, detail=f"Invalid content type: {file.content_type}. Expected video/mp4")


def save_video_meta(video_id: str, meta: dict) -> None:
    _save_json(VIDEOS_DIR / f"{video_id}_meta.json", {"video_id": video_id, **meta})


def save_detections(video_id: str, detections: List[Detection]) -> None:
    _save_json(TRACKS_DIR / f"{video_id}.json", [d.model_dump() for d in detections])


def load_detections(video_id: str) -> Optional[List[Detection]]:
    data = _load_json(TRACKS_DIR / f"{video_id}.json")
    return [Detection(**d) for d in data] if data is not None else None


def save_homographies(video_id: str, h_dict: Dict[int, np.ndarray]) -> None:
    _save_json(ANNOTATIONS_DIR / f"{video_id}_homographies.json", _serialize_H(h_dict))


def load_homographies(video_id: str) -> Optional[Dict[int, np.ndarray]]:
    data = _load_json(ANNOTATIONS_DIR / f"{video_id}_homographies.json")
    return _deserialize_H(data) if data is not None else None


def save_anchor_homographies(video_id: str, h_dict: Dict[int, Any]) -> None:
    _save_json(ANNOTATIONS_DIR / f"{video_id}_anchor_homographies.json", _serialize_H(h_dict))


def load_anchor_homographies(video_id: str) -> Optional[Dict[int, Any]]:
    data = _load_json(ANNOTATIONS_DIR / f"{video_id}_anchor_homographies.json")
    return _deserialize_H(data) if data is not None else None


def save_v3_anchor_homographies(video_id: str, h_dict: Dict[int, Any]) -> None:
    _save_json(ANNOTATIONS_DIR / f"{video_id}_v3_anchor_homographies.json", _serialize_H(h_dict))


def load_v3_anchor_homographies(video_id: str) -> Optional[Dict[int, Any]]:
    data = _load_json(ANNOTATIONS_DIR / f"{video_id}_v3_anchor_homographies.json")
    return _deserialize_H(data) if data is not None else None


def save_v3_homographies(video_id: str, h_dict: Dict[int, np.ndarray]) -> None:
    _save_json(ANNOTATIONS_DIR / f"{video_id}_v3_homographies.json", _serialize_H(h_dict))


def load_v3_homographies(video_id: str) -> Optional[Dict[int, np.ndarray]]:
    data = _load_json(ANNOTATIONS_DIR / f"{video_id}_v3_homographies.json")
    return _deserialize_H(data) if data is not None else None



def _serialise_ann_value(obj) -> Any:
    """Convert a PitchPoint/LineAnnotation or dict to a JSON-serialisable dict."""
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    return obj


def save_annotations(video_id: str, annotations_dict: dict) -> None:
    serialisable = {
        str(frame_idx): {
            "keypoints": [_serialise_ann_value(p) for p in ann.get("keypoints", [])],
            "lines": [_serialise_ann_value(ln) for ln in ann.get("lines", [])],
        }
        for frame_idx, ann in annotations_dict.items()
    }
    _save_json(ANNOTATIONS_DIR / f"{video_id}_annotations.json", serialisable)


def load_annotations(video_id: str) -> Optional[dict]:
    data = _load_json(ANNOTATIONS_DIR / f"{video_id}_annotations.json")
    return {int(k): v for k, v in data.items()} if data is not None else None


def _resolve_homography(video_id: str, frame_idx: int) -> Optional[np.ndarray]:
    """Return the best available homography for a frame (v3 preferred, v2 fallback)."""
    v3_hs = store.v3_homographies_cache.get(video_id) or load_v3_homographies(video_id)
    if v3_hs:
        return v3_hs[frame_idx] if frame_idx in v3_hs else v3_hs[min(v3_hs.keys(), key=lambda f: abs(f - frame_idx))]
    H, _ = find_nearest_homography(video_id, frame_idx, method="v2")
    return H


# Reference lines for warped frame overlays: (label, y_meters, bgr_colour)
_REFERENCE_LINES = [
    ("13m",     13.0, (0, 200, 255)),
    ("20m",     20.0, (0, 255, 0)),
    ("45m",     45.0, (255, 128, 0)),
    ("65m",     65.0, (128, 0, 255)),
    ("halfway", 70.0, (255, 255, 0)),
]


def _draw_reference_lines(warped: np.ndarray) -> None:
    """Draw horizontal pitch reference lines onto a warped canvas image (in-place)."""
    for label, y_m, colour in _REFERENCE_LINES:
        y_px = int(y_m / GAA_PITCH_LENGTH * OUT_H)
        x, dash_len = 0, 20
        while x < OUT_W:
            cv2.line(warped, (x, y_px), (min(x + dash_len, OUT_W), y_px), colour, 2)
            x += dash_len * 2
        cv2.putText(warped, label, (4, y_px - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1, cv2.LINE_AA)


def _read_and_warp_frame(video_path: str, frame_idx: int, H: np.ndarray) -> Response:
    """Extract a video frame, apply homography warp, return JPEG Response."""
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise HTTPException(status_code=500, detail="Failed to extract frame")
    warped = warp_frame(frame, H, OUT_W, OUT_H)
    _, buffer = cv2.imencode('.jpg', warped, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return Response(content=buffer.tobytes(), media_type="image/jpeg", headers={"Cache-Control": "max-age=3600"})


def find_nearest_homography(video_id: str, frame_idx: int, method: str = "v2"):
    """Return (H_matrix, anchor_frame_idx) for the nearest computed anchor frame.

    Args:
        video_id:  Video identifier.
        frame_idx: Target frame index.
        method:    ``"v2"`` (default) or ``"v3"`` — selects which per-frame
                   homography set to use.
    """
    if method == "v3":
        homographies = store.v3_homographies_cache.get(video_id) or load_v3_homographies(video_id)
    else:
        homographies = store.homographies_cache.get(video_id) or load_homographies(video_id)
    if not homographies:
        return None, None
    nearest = min(homographies.keys(), key=lambda f: abs(f - frame_idx))
    return homographies[nearest], nearest


# --- Endpoints ---

@app.post("/videos", response_model=VideoCreateResponse)
async def upload_video(file: UploadFile = File(...)):
    """Upload a video file and extract metadata."""
    content = await file.read()
    validate_video_upload(file, content)

    video_id = str(uuid.uuid4())
    video_path = VIDEOS_DIR / f"{video_id}.mp4"
    with open(video_path, "wb") as f:
        f.write(content)

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


@app.get("/videos/{video_id}/frame/{frame_idx}")
async def get_frame(video_id: str, frame_idx: int):
    """Extract and return a single video frame as JPEG."""
    video_info = _get_video_or_404(video_id)

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


@app.get("/videos/{video_id}/frames/{frame_idx}/warped")
async def get_warped_frame_any(
    video_id: str,
    frame_idx: int,
    players: bool = Query(False, description="Overlay player positions on the warped frame"),
):
    """Return a warped JPEG with pitch reference lines, and optionally player dots.

    Always uses the best available homography (v3 per-frame if computed, else v2
    nearest anchor). Add ``?players=true`` to overlay player positions.
    """
    video_info = _get_video_or_404(video_id)

    H = _resolve_homography(video_id, frame_idx)
    if H is None:
        raise HTTPException(status_code=400, detail="No homographies computed for this video")

    try:
        cap = cv2.VideoCapture(video_info["path"])
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            raise HTTPException(status_code=500, detail="Failed to extract frame")

        warped = warp_frame(frame, H, OUT_W, OUT_H)
        _draw_reference_lines(warped)

        if players:
            for pos in (p for p in store.player_positions_cache.get(video_id, []) if p.frame_idx == frame_idx):
                x, y = int(pos.x_pitch), int(pos.y_pitch)
                if 0 <= x < OUT_W and 0 <= y < OUT_H:
                    color = (0, 0, 255) if pos.track_id % 2 == 0 else (255, 0, 0)
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


@app.get("/videos/{video_id}/detections", response_model=List[Detection])
async def get_video_detections(video_id: str):
    """Return the raw YOLO+BotSort detections for a video."""
    _get_video_or_404(video_id)

    detections = store.detections_cache.get(video_id) or load_detections(video_id)
    if detections is None:
        raise HTTPException(status_code=404, detail="No detections found. Run tracking first.")
    return detections


@app.get("/videos/{video_id}/frames/{frame_idx}/detections_overlay")
async def get_detections_overlay(video_id: str, frame_idx: int):
    """Return a JPEG of the raw video frame with BotSort bounding boxes overlaid."""
    video_info = _get_video_or_404(video_id)

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
            # Unique colour per track ID (same HSV hue trick as the pitch canvas)
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


@app.post("/videos/{video_id}/track", response_model=TrackResponse)
async def track_video(video_id: str):
    """Run YOLO + BotSort tracking on the video."""
    video_info = _get_video_or_404(video_id)
    video_path = video_info["path"]

    detections = load_detections(video_id)
    if detections is None:
        try:
            from pipeline.detect import run_tracking
            logger.info(f"Running tracking on video {video_id}")
            detections = run_tracking(video_path)
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


@app.post("/videos/{video_id}/homographies/v2")
async def compute_homographies_v2(
    video_id: str,
    annotations: List[AnchorFrameAnnotation],
    num_samples_per_line: int = Query(10, ge=2, le=50, description="Points to sample per line"),
    max_iterations: int = Query(3, ge=1, le=10, description="Maximum refinement iterations"),
    keypoint_weight: int = Query(3, ge=1, le=10, description="Weight multiplier for keypoints vs line points"),
):
    """
    Compute homography matrices with line constraint support.

    Accepts both keypoint and line annotations per anchor frame. Line annotations
    provide additional constraints for regions where point intersections are not
    visible (e.g., midfield). Propagates anchor homographies to every frame via
    ORB feature matching.

    Available line IDs: 13m_top, 20m_top, 45m_top, 65m_top, halfway,
    65m_bottom, 45m_bottom, 20m_bottom, 13m_bottom.
    """
    video_info = _get_video_or_404(video_id)

    annotations_dict = {
        ann.frame_idx: {"keypoints": ann.points, "lines": ann.lines}
        for ann in annotations
    }

    try:
        anchor_homographies, computation_info = await asyncio.to_thread(
            partial(
                compute_homographies_with_lines,
                annotations_dict,
                num_samples_per_line=num_samples_per_line,
                max_iterations=max_iterations,
                keypoint_weight=keypoint_weight,
            )
        )
    except Exception as e:
        logger.error(f"Line-constrained homography computation failed for video {video_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Homography computation failed: {str(e)}")

    if not anchor_homographies:
        raise HTTPException(status_code=400, detail="No valid homographies computed. Need at least 4 keypoints per frame.")

    store.anchor_homographies_cache[video_id] = anchor_homographies
    save_anchor_homographies(video_id, anchor_homographies)
    save_annotations(video_id, annotations_dict)

    try:
        per_frame_hs, _ = await asyncio.to_thread(
            partial(
                build_constrained_per_frame_H,
                video_info["path"],
                anchor_homographies,
                start_frame=0,
                end_frame=video_info["num_frames"] - 1,
            )
        )
    except Exception as e:
        logger.error(f"Per-frame H propagation failed for video {video_id}: {e}")
        per_frame_hs = anchor_homographies

    store.homographies_cache[video_id] = per_frame_hs
    save_homographies(video_id, per_frame_hs)

    total_lines_used = sum(info.get('valid_lines', 0) for info in computation_info.values())
    logger.info(
        f"Computed {len(anchor_homographies)} anchor Hs + {len(per_frame_hs)} per-frame Hs "
        f"for video {video_id} using {total_lines_used} line constraints"
    )

    return {
        "frames": sorted(anchor_homographies.keys()),
        "per_frame_count": len(per_frame_hs),
        "info": {str(k): v for k, v in computation_info.items()},
    }


@app.post("/videos/{video_id}/homographies/v3")
async def compute_homographies_v3(
    video_id: str,
    annotations: List[AnchorFrameAnnotation],
    num_samples_per_line: int = Query(10, ge=2, le=50, description="Points to sample per line"),
    ransac_iterations: int = Query(2000, ge=100, le=10000, description="RANSAC trials for keypoint-only H₀"),
    ransac_threshold: float = Query(5.0, ge=1.0, le=50.0, description="RANSAC inlier threshold in canvas pixels"),
    keypoint_weight: float = Query(20.0, ge=1.0, le=100.0, description="Weight multiplier for keypoints vs line samples (higher = lines reinforce more gently)"),
):
    """
    Compute homography matrices using genuine DLT line constraints (v3).

    Unlike v2, each line annotation provides one-dimensional constraints
    directly in the DLT system (one row per sample point) rather than
    synthetic full-point correspondences that circularly depend on the
    current H estimate.

    Results are stored separately from v2 and do not overwrite them.
    Per-frame Hs are propagated via Lucas-Kanade optical flow with forward-backward
    consistency, bidirectional drift correction, and Savitzky-Golay smoothing.
    """
    from pipeline.homography import compute_homographies_with_lines_v3

    video_info = _get_video_or_404(video_id)

    annotations_dict = {
        ann.frame_idx: {"keypoints": ann.points, "lines": ann.lines}
        for ann in annotations
    }

    try:
        anchor_homographies, computation_info = await asyncio.to_thread(
            partial(
                compute_homographies_with_lines_v3,
                annotations_dict,
                num_samples_per_line=num_samples_per_line,
                ransac_iterations=ransac_iterations,
                ransac_threshold=ransac_threshold,
                keypoint_weight=keypoint_weight,
            )
        )
    except Exception as e:
        logger.error(f"v3 DLT homography computation failed for video {video_id}: {e}")
        raise HTTPException(status_code=500, detail=f"v3 homography computation failed: {str(e)}")

    if not anchor_homographies:
        raise HTTPException(
            status_code=400,
            detail="No valid v3 homographies computed. Ensure at least one frame has line annotations."
        )

    store.v3_anchor_homographies_cache[video_id] = anchor_homographies
    save_v3_anchor_homographies(video_id, anchor_homographies)
    save_annotations(video_id, annotations_dict)

    try:
        per_frame_hs, of_info = await asyncio.to_thread(
            partial(
                build_optical_flow_per_frame_H,
                video_info["path"],
                anchor_homographies,
                total_frames=video_info["num_frames"],
            )
        )
        logger.info(
            f"v3 optical flow: {of_info.get('num_frames')} frames, "
            f"{len(of_info.get('failed_frames', []))} failed OF pairs, "
            f"drift norms: {of_info.get('drift_at_anchors')}"
        )
    except Exception as e:
        logger.error(f"v3 per-frame H propagation failed for video {video_id}: {e}")
        per_frame_hs = anchor_homographies

    store.v3_homographies_cache[video_id] = per_frame_hs
    save_v3_homographies(video_id, per_frame_hs)

    logger.info(
        f"v3 DLT: computed {len(anchor_homographies)} anchor Hs + {len(per_frame_hs)} per-frame Hs "
        f"for video {video_id}"
    )

    return {
        "frames": sorted(anchor_homographies.keys()),
        "per_frame_count": len(per_frame_hs),
        "info": {str(k): v for k, v in computation_info.items()},
    }


@app.get("/line-constraints/available-lines")
async def get_available_line_ids():
    """Get available line IDs and their Y positions (meters) for line annotations."""
    from pipeline.line_constraints import GAA_PITCH_LINES
    return {
        "lines": GAA_PITCH_LINES,
        "description": {
            "13m_top": "13 meter line (top/near goal)",
            "20m_top": "20 meter line (top)",
            "45m_top": "45 meter line (top)",
            "65m_top": "65 meter line (top)",
            "halfway": "Halfway line (70m)",
            "65m_bottom": "65 meter line (bottom)",
            "45m_bottom": "45 meter line (bottom)",
            "20m_bottom": "20 meter line (bottom)",
            "13m_bottom": "13 meter line (bottom/far goal)",
        },
    }


@app.post("/videos/{video_id}/map_players", response_model=List[PlayerPitchPosition])
async def map_players(video_id: str):
    """Map player detections to pitch coordinates using computed homographies."""
    _get_video_or_404(video_id)

    detections = load_detections(video_id)
    if detections is None:
        raise HTTPException(status_code=400, detail="No detections found. Run tracking first.")

    homographies = (
        store.v3_homographies_cache.get(video_id) or load_v3_homographies(video_id) or
        store.homographies_cache.get(video_id) or load_homographies(video_id)
    )
    if homographies is None:
        raise HTTPException(status_code=400, detail="No homographies found. Compute homographies first.")

    anchor_hs = (
        store.v3_anchor_homographies_cache.get(video_id) or load_v3_anchor_homographies(video_id) or
        store.anchor_homographies_cache.get(video_id) or load_anchor_homographies(video_id)
    )
    anchor_frame_indices = set(anchor_hs.keys()) if anchor_hs else None

    try:
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




@app.get("/videos/{video_id}/homographies/anchor-quality")
async def get_anchor_quality(video_id: str):
    """
    Compute per-keypoint reprojection error for each anchor frame.

    Works with both v2 and v3 anchor homographies (v2 takes priority if both exist).
    """
    _get_video_or_404(video_id)

    annotations = load_annotations(video_id)
    if annotations is None:
        raise HTTPException(status_code=400, detail="No annotations found. Compute homographies first.")

    anchor_hs = (
        store.anchor_homographies_cache.get(video_id)
        or load_anchor_homographies(video_id)
        or store.v3_anchor_homographies_cache.get(video_id)
        or load_v3_anchor_homographies(video_id)
    )
    if not anchor_hs:
        raise HTTPException(status_code=400, detail="No anchor homographies found. Compute homographies first.")

    def _to_canvas(pitch_id: str):
        x_m, y_m = resolve_pitch_coordinates(pitch_id)
        return x_m / GAA_PITCH_WIDTH * OUT_W, y_m / GAA_PITCH_LENGTH * OUT_H

    anchors = []
    for frame_idx in sorted(anchor_hs.keys()):
        H = anchor_hs[frame_idx]
        ann = annotations.get(frame_idx, {})
        keypoints_raw = ann.get("keypoints", [])
        lines_raw = ann.get("lines", [])

        kp_results = []
        for kp in keypoints_raw:
            if isinstance(kp, dict):
                pitch_id, x_img, y_img = kp.get("pitch_id", ""), float(kp.get("x_img", 0)), float(kp.get("y_img", 0))
            else:
                pitch_id, x_img, y_img = getattr(kp, "pitch_id", ""), float(getattr(kp, "x_img", 0)), float(getattr(kp, "y_img", 0))

            p = np.array([x_img, y_img, 1.0], dtype=np.float64)
            projected = H @ p
            if abs(projected[2]) > 1e-12:
                projected /= projected[2]
            x_pred, y_pred = float(projected[0]), float(projected[1])

            try:
                x_exp, y_exp = _to_canvas(pitch_id)
            except ValueError:
                continue

            error_px = float(np.sqrt((x_pred - x_exp) ** 2 + (y_pred - y_exp) ** 2))
            if error_px > 30:
                verdict, impact = "outlier", "harmful"
            elif error_px > 15:
                verdict, impact = "high", "marginal"
            else:
                verdict, impact = "good", "helpful"

            kp_results.append({
                "pitch_id": pitch_id, "x_img": x_img, "y_img": y_img,
                "error_px": round(error_px, 2), "verdict": verdict, "impact": impact,
            })

        if not kp_results:
            continue

        errors = [kp["error_px"] for kp in kp_results]
        mean_err = float(np.mean(errors))
        max_err = float(np.max(errors))
        n_outliers = sum(1 for kp in kp_results if kp["verdict"] == "outlier")
        n_helpful = sum(1 for kp in kp_results if kp["impact"] == "helpful")

        if n_outliers > 0 or mean_err > 30:
            overall = "bad"
        elif mean_err > 15:
            overall = "warning"
        else:
            overall = "good"

        recommendation = (
            f"Remove or re-annotate {n_outliers} outlier point(s) listed above." if n_outliers > 0
            else "Consider improving marginal keypoints for better accuracy." if overall == "warning"
            else "Homography quality looks good."
        )

        anchors.append({
            "frame_idx": frame_idx,
            "n_keypoints": len(kp_results),
            "n_lines": len(lines_raw),
            "mean_error_px": round(mean_err, 2),
            "max_error_px": round(max_err, 2),
            "n_outlier_points": n_outliers,
            "n_helpful_points": n_helpful,
            "overall_quality": overall,
            "keypoints": kp_results,
            "recommendation": recommendation,
        })

    return {"anchors": anchors}


@app.post("/videos/{video_id}/interpolate", response_model=InterpolationResponse)
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
    _get_video_or_404(video_id)

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


@app.get("/videos/{video_id}/players", response_model=List[PlayerPitchPosition])
async def get_player_positions(video_id: str):
    """Get all player positions (sparse + interpolated) for a video."""
    _get_video_or_404(video_id)

    positions = store.player_positions_cache.get(video_id)
    if positions is None:
        raise HTTPException(status_code=404, detail="No player positions found. Run map_players first.")

    return sorted(positions, key=lambda p: (p.frame_idx, p.track_id))


@app.get("/videos/{video_id}/diagnostics/per-frame-mapping")
async def diagnostics_per_frame_mapping(video_id: str):
    """
    Compute per-frame mapping accuracy using annotated 20m_top line points as
    simulated players with a known expected canvas Y position.

    Compares per-frame propagated H vs nearest anchor H accuracy.
    Requires that /homographies/v2 has been run first.
    """
    _get_video_or_404(video_id)

    annotations = load_annotations(video_id)
    if annotations is None:
        raise HTTPException(status_code=400, detail="No annotations found. Compute homographies (v2) first.")

    anchor_hs = store.anchor_homographies_cache.get(video_id) or load_anchor_homographies(video_id)
    if not anchor_hs:
        raise HTTPException(status_code=400, detail="No anchor homographies found. Compute homographies (v2) first.")

    per_frame_hs = store.homographies_cache.get(video_id) or load_homographies(video_id)
    if not per_frame_hs:
        raise HTTPException(status_code=400, detail="No per-frame homographies found. Compute homographies (v2) first.")

    from pipeline.homography import map_pixel_to_pitch

    expected_y_20m = 20.0 / GAA_PITCH_LENGTH * OUT_H
    anchor_list = sorted(anchor_hs.keys())
    results = []

    for idx_in_list, anchor_frame_idx in enumerate(anchor_list):
        ann = annotations.get(anchor_frame_idx, {})
        line_20m = next((l for l in ann.get("lines", []) if l.get("line_id") == "20m_top"), None)
        if line_20m is None:
            continue

        mid_img_x = (line_20m["u1"] + line_20m["u2"]) / 2.0
        mid_img_y = (line_20m["v1"] + line_20m["v2"]) / 2.0
        next_anchor = anchor_list[idx_in_list + 1] if idx_in_list + 1 < len(anchor_list) else anchor_frame_idx + 1

        for f in range(anchor_frame_idx, next_anchor):
            H_pf = per_frame_hs.get(f)
            if H_pf is None:
                continue

            nearest = min(anchor_list, key=lambda a: abs(a - f))
            H_na = anchor_hs[nearest]

            _, y_pf = map_pixel_to_pitch(mid_img_x, mid_img_y, H_pf)
            _, y_na = map_pixel_to_pitch(mid_img_x, mid_img_y, H_na)

            err_pf = abs(y_pf - expected_y_20m)
            err_na = abs(y_na - expected_y_20m)

            results.append({
                "frame": f,
                "y_pf": round(float(y_pf), 2),
                "y_na": round(float(y_na), 2),
                "err_pf": round(float(err_pf), 2),
                "err_na": round(float(err_na), 2),
                "improvement": round(float(err_na - err_pf), 2),
            })

    if not results:
        return {"frames": [], "summary": None}

    errs_pf = [r["err_pf"] for r in results]
    errs_na = [r["err_na"] for r in results]
    pct_better = 100.0 * sum(1 for r in results if r["improvement"] > 2) / len(results)

    return {
        "frames": results,
        "summary": {
            "median_pf": round(float(np.median(errs_pf)), 2),
            "mean_pf": round(float(np.mean(errs_pf)), 2),
            "max_pf": round(float(np.max(errs_pf)), 2),
            "median_na": round(float(np.median(errs_na)), 2),
            "mean_na": round(float(np.mean(errs_na)), 2),
            "max_na": round(float(np.max(errs_na)), 2),
            "pct_better": round(pct_better, 1),
        },
    }
