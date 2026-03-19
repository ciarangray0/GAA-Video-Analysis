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
from dotenv import load_dotenv
load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")

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
    TeamOverrideRequest,
    VALID_TEAMS,
)
# NOTE: `run_tracking` performs heavy ML imports; imported lazily inside endpoint.
from pipeline.homography import resolve_pitch_coordinates
from pipeline.constrained_homography import build_optical_flow_per_frame_H
from pipeline.map_players import map_players_to_pitch, filter_detections_for_mapping
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


def _save_homography_dict(video_id: str, key: str, h_dict: Dict[int, np.ndarray]) -> None:
    _save_json(ANNOTATIONS_DIR / f"{video_id}_{key}.json", _serialize_H(h_dict))


def _load_homography_dict(video_id: str, key: str) -> Optional[Dict[int, np.ndarray]]:
    data = _load_json(ANNOTATIONS_DIR / f"{video_id}_{key}.json")
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


def _load_team_classifications(video_id: str) -> Optional[Dict[int, dict]]:
    data = _load_json(ANNOTATIONS_DIR / f"{video_id}_team_classifications.json")
    return {int(k): v for k, v in data.items()} if data is not None else None


def _resolve_homography(video_id: str, frame_idx: int) -> Optional[np.ndarray]:
    """Return the per-frame v3 homography for a frame (nearest if exact frame missing)."""
    v3_hs = store.v3_per_frame_H_cache.get(video_id) or _load_homography_dict(video_id, "v3_homographies")
    if not v3_hs:
        return None
    if frame_idx in v3_hs:
        return v3_hs[frame_idx]
    return v3_hs[min(v3_hs.keys(), key=lambda f: abs(f - frame_idx))]


# Reference lines for warped frame overlays: (label, y_meters, bgr_colour)
_REFERENCE_LINES = [
    ("13m",     13.0, (0, 200, 255)),
    ("20m",     20.0, (0, 255, 0)),
    ("45m",     45.0, (255, 128, 0)),
    ("65m",     65.0, (128, 0, 255)),
    ("halfway", 70.0, (255, 255, 0)),
    ("65m",     75.0, (128, 0, 255)),
    ("45m",     95.0, (255, 128, 0)),
    ("20m",    120.0, (0, 255, 0)),
    ("13m",    127.0, (0, 200, 255)),
]


_LINE_ALPHA = 0.45  # overlay opacity for pitch reference lines


def _draw_reference_lines(warped: np.ndarray) -> None:
    """Draw pitch reference lines onto a warped canvas image (in-place, semi-transparent)."""
    overlay = warped.copy()

    # Horizontal reference lines (dashed)
    for label, y_m, colour in _REFERENCE_LINES:
        y_px = int(y_m / GAA_PITCH_LENGTH * OUT_H)
        x, dash_len = 0, 20
        while x < OUT_W:
            cv2.line(overlay, (x, y_px), (min(x + dash_len, OUT_W), y_px), colour, 2)
            x += dash_len * 2
        cv2.putText(overlay, label, (4, y_px - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1, cv2.LINE_AA)

    # 20m semicircles (radius 13m = 130px, curving into pitch)
    arc_r = int(13 / GAA_PITCH_LENGTH * OUT_H)
    arc_cx = OUT_W // 2
    colour_20m = (0, 255, 0)
    cv2.ellipse(overlay, (arc_cx, int(20 / GAA_PITCH_LENGTH * OUT_H)), (arc_r, arc_r), 0, 0, 180, colour_20m, 2)
    cv2.ellipse(overlay, (arc_cx, int(120 / GAA_PITCH_LENGTH * OUT_H)), (arc_r, arc_r), 0, 180, 360, colour_20m, 2)

    # 13m box vertical lines (x=33m, x=52m, from endline to 13m line)
    box_colour = (255, 255, 255)
    box13_lx = int(33 / GAA_PITCH_WIDTH * OUT_W)
    box13_rx = int(52 / GAA_PITCH_WIDTH * OUT_W)
    y13_top = int(13 / GAA_PITCH_LENGTH * OUT_H)
    y13_bot = int(127 / GAA_PITCH_LENGTH * OUT_H)
    cv2.line(overlay, (box13_lx, 0),      (box13_lx, y13_top), box_colour, 2)
    cv2.line(overlay, (box13_rx, 0),      (box13_rx, y13_top), box_colour, 2)
    cv2.line(overlay, (box13_lx, y13_bot), (box13_lx, OUT_H),  box_colour, 2)
    cv2.line(overlay, (box13_rx, y13_bot), (box13_rx, OUT_H),  box_colour, 2)

    # Small (goalie) box (x=35.5m–49.5m, depth=4.5m from each endline)
    boxs_lx = int(35.5 / GAA_PITCH_WIDTH * OUT_W)
    boxs_rx = int(49.5 / GAA_PITCH_WIDTH * OUT_W)
    ys_top = int(4.5 / GAA_PITCH_LENGTH * OUT_H)
    ys_bot = int(135.5 / GAA_PITCH_LENGTH * OUT_H)
    cv2.line(overlay, (boxs_lx, 0),      (boxs_lx, ys_top),  box_colour, 2)
    cv2.line(overlay, (boxs_lx, ys_top), (boxs_rx, ys_top),  box_colour, 2)
    cv2.line(overlay, (boxs_rx, 0),      (boxs_rx, ys_top),  box_colour, 2)
    cv2.line(overlay, (boxs_lx, OUT_H),  (boxs_lx, ys_bot),  box_colour, 2)
    cv2.line(overlay, (boxs_lx, ys_bot), (boxs_rx, ys_bot),  box_colour, 2)
    cv2.line(overlay, (boxs_rx, OUT_H),  (boxs_rx, ys_bot),  box_colour, 2)

    cv2.addWeighted(overlay, _LINE_ALPHA, warped, 1 - _LINE_ALPHA, 0, warped)


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
            classifications = (
                store.team_classifications_cache.get(video_id)
                or _load_team_classifications(video_id)
                or {}
            )
            for pos in (p for p in store.player_positions_cache.get(video_id, []) if p.frame_idx == frame_idx):
                team = classifications.get(pos.track_id, {}).get("team", "")
                if team in ("referee", "ignore"):
                    continue
                x, y = int(pos.x_pitch), int(pos.y_pitch)
                if 0 <= x < OUT_W and 0 <= y < OUT_H:
                    if team == "ellistown":
                        color = (0, 210, 255)   # yellow in BGR
                    elif team == "opposition":
                        color = (220, 80, 50)   # blue in BGR
                    else:
                        color = (0, 0, 255)     # default red (unclassified)
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
    Compute anchor homographies using DLT line constraints, then propagate
    per-frame via Lucas-Kanade optical flow with drift correction and SG smoothing.

    Each line annotation provides one-dimensional constraints directly in the
    DLT system (one row per sample point).
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
        frame_errors = {
            str(k): v.get("error", "unknown") for k, v in computation_info.items()
        }
        logger.warning(f"v3: no valid homographies for {video_id}. Per-frame: {frame_errors}")
        raise HTTPException(
            status_code=400,
            detail=(
                "No valid v3 homographies computed. Each annotated frame needs at least 4 keypoints. "
                f"Per-frame errors: {frame_errors}"
            ),
        )

    store.v3_anchor_H_cache[video_id] = anchor_homographies
    _save_homography_dict(video_id, "v3_anchor_homographies", anchor_homographies)
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

    store.v3_per_frame_H_cache[video_id] = per_frame_hs
    _save_homography_dict(video_id, "v3_homographies", per_frame_hs)

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
    from pipeline.gaa_pitch_config import GAA_PITCH_LINES
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

    homographies = store.v3_per_frame_H_cache.get(video_id) or _load_homography_dict(video_id, "v3_homographies")
    if homographies is None:
        raise HTTPException(status_code=400, detail="No homographies found. Compute homographies first.")

    anchor_hs = store.v3_anchor_H_cache.get(video_id) or _load_homography_dict(video_id, "v3_anchor_homographies")
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




@app.get("/videos/{video_id}/homographies/anchor-quality")
async def get_anchor_quality(video_id: str):
    """
    Compute per-keypoint reprojection error for each anchor frame.

    """
    _get_video_or_404(video_id)

    annotations = load_annotations(video_id)
    if annotations is None:
        raise HTTPException(status_code=400, detail="No annotations found. Compute homographies first.")

    anchor_hs = store.v3_anchor_H_cache.get(video_id) or _load_homography_dict(video_id, "v3_anchor_homographies")
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


# --- Team Classification ---

@app.post("/videos/{video_id}/classify-teams")
async def classify_teams(video_id: str):
    """Classify player tracks into teams by jersey colour analysis.

    Samples video frames for each track, extracts the jersey HSV colour,
    and assigns 'ellistown' (yellow jersey) or 'opposition'.
    Returns per-track classifications plus a summary with cluster statistics.
    """
    from pipeline.team_classifier import classify_tracks

    video_info = _get_video_or_404(video_id)

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
    _save_json(
        ANNOTATIONS_DIR / f"{video_id}_team_classifications.json",
        {str(k): v for k, v in classifications.items()},
    )

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


@app.get("/videos/{video_id}/classify-teams")
async def get_team_classifications(video_id: str):
    """Return stored team classifications for a video."""
    _get_video_or_404(video_id)

    classifications = (
        store.team_classifications_cache.get(video_id)
        or _load_team_classifications(video_id)
    )
    if classifications is None:
        raise HTTPException(status_code=404, detail="No team classifications found. Run classify-teams first.")

    return {"classifications": {str(k): v for k, v in classifications.items()}}


@app.patch("/videos/{video_id}/classify-teams")
async def override_team_classification(video_id: str, body: TeamOverrideRequest):
    """Override a single track's team assignment."""
    from pipeline.team_classifier import override_classification

    _get_video_or_404(video_id)

    if body.team not in VALID_TEAMS:
        raise HTTPException(
            status_code=400,
            detail=f"team must be one of: {', '.join(sorted(VALID_TEAMS))}",
        )

    classifications = (
        store.team_classifications_cache.get(video_id)
        or _load_team_classifications(video_id)
        or {}
    )

    classifications = override_classification(classifications, body.track_id, body.team)
    store.team_classifications_cache[video_id] = classifications
    _save_json(
        ANNOTATIONS_DIR / f"{video_id}_team_classifications.json",
        {str(k): v for k, v in classifications.items()},
    )

    logger.info(f"Track {body.track_id} reassigned to '{body.team}' for video {video_id}")
    return {"classifications": {str(k): v for k, v in classifications.items()}}


