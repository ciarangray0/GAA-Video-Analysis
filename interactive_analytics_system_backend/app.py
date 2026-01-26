"""FastAPI application for video analysis pipeline."""
import os
import uuid
from pathlib import Path
from typing import List, Dict, Optional
import json
import numpy as np
import logging

from fastapi import FastAPI, UploadFile, File, HTTPException, Query, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

from pipeline.schemas import (
    VideoCreateResponse,
    PitchAnnotation,
    TrackResponse,
    HomographyResponse,
    InterpolationResponse,
    Detection,
    PlayerPitchPosition,
    ProcessVideoResponse,
    LineAnnotation,
    AnchorFrameAnnotation
)
# NOTE: `run_tracking` performs heavy ML imports; import lazily inside endpoints to keep module import lightweight.
# from pipeline.detect import run_tracking
from pipeline.homography import compute_homographies_from_annotations, compute_homographies_with_lines
from pipeline.map_players import map_players_to_pitch
from pipeline.trajectories import interpolate_trajectories
from pipeline.video import get_video_metadata, extract_frame

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration from environment variables
MAX_VIDEO_SIZE_MB = int(os.getenv("MAX_VIDEO_SIZE_MB", "500"))
MAX_VIDEO_SIZE = MAX_VIDEO_SIZE_MB * 1024 * 1024  # Convert to bytes
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")
DATA_DIR = Path(os.getenv("DATA_DIR", "data"))

app = FastAPI(title="GAA Video Analysis API")

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Data directories
VIDEOS_DIR = DATA_DIR / "videos"
TRACKS_DIR = DATA_DIR / "tracks"
ANNOTATIONS_DIR = DATA_DIR / "annotations"

# Create directories if they don't exist
VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
TRACKS_DIR.mkdir(parents=True, exist_ok=True)
ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)

# In-memory storage (could be replaced with database)
videos: Dict[str, dict] = {}
detections_cache: Dict[str, List[Detection]] = {}
homographies_cache: Dict[str, Dict[int, any]] = {}
player_positions_cache: Dict[str, List[PlayerPitchPosition]] = {}


# --- Health Check ---
@app.get("/health")
async def health_check():
    """Health check endpoint for container orchestration."""
    return {"status": "ok"}


# --- Helper Functions ---
def validate_video_upload(file: UploadFile, content: bytes) -> None:
    """Validate uploaded video file."""
    # Check file size
    if len(content) > MAX_VIDEO_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size is {MAX_VIDEO_SIZE_MB}MB"
        )

    # Check file extension
    if not file.filename or not file.filename.lower().endswith(".mp4"):
        raise HTTPException(
            status_code=400,
            detail="Only MP4 video files are accepted"
        )

    # Check MIME type if provided
    if file.content_type and file.content_type not in ["video/mp4", "application/octet-stream"]:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid content type: {file.content_type}. Expected video/mp4"
        )


def save_detections(video_id: str, detections: List[Detection]):
    """Save detections to JSON file."""
    file_path = TRACKS_DIR / f"{video_id}.json"
    with open(file_path, "w") as f:
        json.dump([d.model_dump() for d in detections], f, indent=2)


def load_detections(video_id: str) -> Optional[List[Detection]]:
    """Load detections from JSON file."""
    file_path = TRACKS_DIR / f"{video_id}.json"
    if file_path.exists():
        with open(file_path, "r") as f:
            data = json.load(f)
            return [Detection(**d) for d in data]
    return None


def save_homographies(video_id: str, homographies: Dict[int, any]):
    """Save homographies to JSON file (as lists for JSON serialization)."""
    file_path = ANNOTATIONS_DIR / f"{video_id}_homographies.json"
    homographies_serialized = {
        str(k): v.tolist() for k, v in homographies.items()
    }
    with open(file_path, "w") as f:
        json.dump(homographies_serialized, f, indent=2)


def load_homographies(video_id: str) -> Optional[Dict[int, any]]:
    """Load homographies from JSON file."""
    file_path = ANNOTATIONS_DIR / f"{video_id}_homographies.json"
    if file_path.exists():
        with open(file_path, "r") as f:
            data = json.load(f)
            return {int(k): np.array(v) for k, v in data.items()}
    return None


def sanitize_error_message(error: Exception) -> str:
    """Sanitize error messages for client responses."""
    # In production, you may want to log the full error and return a generic message
    error_str = str(error)
    # Remove file paths from error messages
    if "data/" in error_str or "/Users/" in error_str or "/home/" in error_str:
        return "An internal error occurred"
    return error_str


# --- Endpoints ---
@app.post("/videos", response_model=VideoCreateResponse)
async def upload_video(file: UploadFile = File(...)):
    """
    Upload a video file and extract metadata.
    
    Returns video_id, fps, and num_frames.
    """
    # Read file content
    content = await file.read()

    # Validate upload
    validate_video_upload(file, content)

    # Generate unique video ID
    video_id = str(uuid.uuid4())
    
    # Save video file
    video_path = VIDEOS_DIR / f"{video_id}.mp4"
    with open(video_path, "wb") as f:
        f.write(content)
    
    # Extract metadata
    try:
        metadata = get_video_metadata(str(video_path))
    except Exception as e:
        # Clean up file on error
        video_path.unlink()
        logger.error(f"Failed to process video {video_id}: {e}")
        raise HTTPException(status_code=400, detail="Failed to process video. Ensure it is a valid MP4 file.")

    # Store video info
    videos[video_id] = {
        "path": str(video_path),
        "fps": metadata["fps"],
        "num_frames": metadata["num_frames"],
        "width": metadata["width"],
        "height": metadata["height"],
        "duration_seconds": metadata["duration_seconds"]
    }
    
    logger.info(f"Uploaded video {video_id}: {metadata['num_frames']} frames at {metadata['fps']} fps")

    return VideoCreateResponse(
        video_id=video_id,
        fps=metadata["fps"],
        num_frames=metadata["num_frames"],
        width=metadata["width"],
        height=metadata["height"],
        duration_seconds=metadata["duration_seconds"]
    )


@app.get("/videos/{video_id}/frame/{frame_idx}")
async def get_frame(video_id: str, frame_idx: int):
    """
    Extract and return a single frame from a video as JPEG image.

    Args:
        video_id: Video identifier
        frame_idx: Frame index to extract (0-based)

    Returns:
        JPEG image of the frame
    """
    if video_id not in videos:
        raise HTTPException(status_code=404, detail="Video not found")

    video_info = videos[video_id]

    # Validate frame index
    if frame_idx < 0 or frame_idx >= video_info["num_frames"]:
        raise HTTPException(
            status_code=400,
            detail=f"Frame index must be between 0 and {video_info['num_frames'] - 1}"
        )

    try:
        frame_bytes = extract_frame(video_info["path"], frame_idx)
        if frame_bytes is None:
            raise HTTPException(status_code=500, detail="Failed to extract frame")

        return Response(
            content=frame_bytes,
            media_type="image/jpeg",
            headers={"Cache-Control": "max-age=3600"}
        )
    except Exception as e:
        logger.error(f"Failed to extract frame {frame_idx} from video {video_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to extract frame")


@app.get("/videos/{video_id}/warped-frame/{frame_idx}")
async def get_warped_frame(video_id: str, frame_idx: int):
    """
    Extract a frame and warp it using the computed homography.

    This visualizes how the frame maps to the pitch canvas,
    matching the notebook's distorted_homography_warp function.

    Args:
        video_id: Video identifier
        frame_idx: Frame index (must be an anchor frame with homography)

    Returns:
        JPEG image of the warped frame on pitch canvas
    """
    if video_id not in videos:
        raise HTTPException(status_code=404, detail="Video not found")

    video_info = videos[video_id]

    # Check if we have a homography for this frame
    if video_id not in homographies_cache:
        raise HTTPException(status_code=400, detail="No homographies computed for this video")

    homographies = homographies_cache[video_id]
    if frame_idx not in homographies:
        raise HTTPException(
            status_code=400,
            detail=f"No homography for frame {frame_idx}. Available: {list(homographies.keys())}"
        )

    H = homographies[frame_idx]

    try:
        # Import required modules
        import cv2
        import numpy as np
        from pipeline.config import OUT_W, OUT_H, K1

        # Extract the original frame
        cap = cv2.VideoCapture(video_info["path"])
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            raise HTTPException(status_code=500, detail="Failed to extract frame")

        # Warp the frame using the distorted homography (matches notebook)
        warped = distorted_homography_warp(frame, H, OUT_W, OUT_H, K1)

        # Encode as JPEG
        _, buffer = cv2.imencode('.jpg', warped, [cv2.IMWRITE_JPEG_QUALITY, 85])

        return Response(
            content=buffer.tobytes(),
            media_type="image/jpeg",
            headers={"Cache-Control": "max-age=3600"}
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to create warped frame {frame_idx} for video {video_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create warped frame: {str(e)}")


def distorted_homography_warp(img, H, out_w, out_h, k1):
    """
    Warp an image using homography with radial distortion.

    This matches the notebook's distorted_homography_warp function exactly.
    Uses vectorized operations for performance.
    """
    import cv2
    import numpy as np

    h, w = img.shape[:2]
    H_inv = np.linalg.inv(H)

    cx, cy = out_w / 2, out_h / 2

    # Create coordinate grids
    xs, ys = np.meshgrid(np.arange(out_w), np.arange(out_h))

    # Apply radial distortion
    dx = xs - cx
    dy = ys - cy
    r2 = dx**2 + dy**2

    xs_d = xs + dx * k1 * r2
    ys_d = ys + dy * k1 * r2

    # Create homogeneous coordinates
    ones = np.ones_like(xs_d)
    pts_d = np.stack([xs_d, ys_d, ones], axis=-1)

    # Transform back to source image coordinates
    # Reshape for matrix multiplication
    pts_flat = pts_d.reshape(-1, 3)
    src_flat = (H_inv @ pts_flat.T).T
    src_flat = src_flat[:, :2] / src_flat[:, 2:3]

    # Reshape to grid
    map_x = src_flat[:, 0].reshape(out_h, out_w).astype(np.float32)
    map_y = src_flat[:, 1].reshape(out_h, out_w).astype(np.float32)

    # Remap
    warped = cv2.remap(img, map_x, map_y,
                       interpolation=cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_CONSTANT)

    return warped


@app.post("/videos/{video_id}/track", response_model=TrackResponse)
async def track_video(video_id: str):
    """
    Run YOLO + ByteTrack tracking on the video.
    
    Returns number of frames processed and unique tracks detected.
    """
    if video_id not in videos:
        raise HTTPException(status_code=404, detail="Video not found")
    
    video_path = videos[video_id]["path"]
    
    # Check cache first
    detections = load_detections(video_id)
    if detections is None:
        # Run tracking (uses GPU or local based on GPU_PROVIDER env var)
        try:
            from pipeline.detect import run_tracking
            logger.info(f"Running tracking on video {video_id}")
            detections = run_tracking(video_path)
            detections_cache[video_id] = detections
            save_detections(video_id, detections)
        except Exception as e:
            logger.error(f"Tracking failed for video {video_id}: {e}")
            raise HTTPException(status_code=500, detail=f"Tracking failed: {str(e)}")
    else:
        detections_cache[video_id] = detections
    
    if not detections:
        raise HTTPException(status_code=500, detail="No detections found in video")

    # Count unique tracks
    unique_tracks = len(set(d.track_id for d in detections))
    
    # Count frames processed
    frames_processed = max(d.frame_idx for d in detections) + 1
    
    logger.info(f"Tracking complete for {video_id}: {frames_processed} frames, {unique_tracks} tracks")

    return TrackResponse(
        frames_processed=frames_processed,
        tracks=unique_tracks
    )


@app.post("/videos/{video_id}/homographies", response_model=HomographyResponse)
async def compute_homographies(
    video_id: str,
    annotations: List[PitchAnnotation]
):
    """
    Compute homography matrices from pitch annotations.
    
    Accepts a list of PitchAnnotation objects (one per frame).
    Returns list of frame indices for which homographies were computed.
    """
    if video_id not in videos:
        raise HTTPException(status_code=404, detail="Video not found")
    
    # Convert annotations to dict format expected by compute_homographies_from_annotations
    annotations_dict = {}
    for ann in annotations:
        annotations_dict[ann.frame_idx] = ann.points
    
    # Compute homographies
    try:
        homographies = compute_homographies_from_annotations(annotations_dict)
        homographies_cache[video_id] = homographies
        save_homographies(video_id, homographies)
    except Exception as e:
        logger.error(f"Homography computation failed for video {video_id}: {e}")
        raise HTTPException(status_code=500, detail="Homography computation failed")

    if not homographies:
        raise HTTPException(status_code=400, detail="No valid homographies computed. Need at least 4 points per frame.")

    logger.info(f"Computed {len(homographies)} homographies for video {video_id}")

    return HomographyResponse(frames=sorted(homographies.keys()))


# Import BaseModel for the response class (already imported via pydantic in schemas)
from pydantic import BaseModel


class HomographyWithLinesResponse(BaseModel):
    """Response for line-constrained homography computation."""
    frames: List[int]
    info: Dict[str, dict] = {}  # Frame index (as string) -> computation info


@app.post("/videos/{video_id}/homographies/v2")
async def compute_homographies_with_line_constraints(
    video_id: str,
    annotations: List[AnchorFrameAnnotation],
    num_samples_per_line: int = Query(10, ge=2, le=50, description="Points to sample per line"),
    max_iterations: int = Query(3, ge=1, le=10, description="Maximum refinement iterations"),
    keypoint_weight: int = Query(3, ge=1, le=10, description="Weight multiplier for keypoints vs line points")
):
    """
    Compute homography matrices with line constraint support.

    This enhanced endpoint accepts both keypoint and line annotations.
    Line annotations provide additional constraints for regions where
    point intersections are not visible (e.g., midfield).

    **Line Annotation Usage:**
    - Click two points on a known horizontal pitch line (13m, 20m, 45m, 65m, halfway)
    - The system uses the known Y-value of the line to generate synthetic correspondences
    - This improves homography stability in midfield regions

    **Available line IDs:**
    - Top half: "13m_top", "20m_top", "45m_top", "65m_top"
    - Middle: "halfway"
    - Bottom half: "65m_bottom", "45m_bottom", "20m_bottom", "13m_bottom"

    **Algorithm:**
    1. Compute initial homography from keypoints only
    2. For each line, sample points and project through homography
    3. Create synthetic correspondences with known Y, estimated X
    4. Re-compute homography with all points (iteratively)

    Args:
        video_id: Video identifier
        annotations: List of AnchorFrameAnnotation objects with keypoints and lines
        num_samples_per_line: Number of points to sample along each line (default: 10)
        max_iterations: Maximum refinement iterations (default: 3)
        keypoint_weight: How many times more important keypoints are than line points (default: 3)

    Returns:
        frames: List of frame indices with computed homographies
        info: Computation info per frame (iterations, valid_lines, warnings, etc.)
    """
    if video_id not in videos:
        raise HTTPException(status_code=404, detail="Video not found")

    # Convert annotations to dict format
    annotations_dict = {}
    for ann in annotations:
        annotations_dict[ann.frame_idx] = {
            "keypoints": ann.points,
            "lines": ann.lines
        }

    # Compute with line constraints
    try:
        homographies, computation_info = compute_homographies_with_lines(
            annotations_dict,
            num_samples_per_line=num_samples_per_line,
            max_iterations=max_iterations,
            keypoint_weight=keypoint_weight
        )
        homographies_cache[video_id] = homographies
        save_homographies(video_id, homographies)
    except Exception as e:
        logger.error(f"Line-constrained homography computation failed for video {video_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Homography computation failed: {str(e)}")

    if not homographies:
        raise HTTPException(
            status_code=400,
            detail="No valid homographies computed. Need at least 4 keypoints per frame."
        )

    # Log summary
    total_lines_used = sum(info.get('valid_lines', 0) for info in computation_info.values())
    logger.info(
        f"Computed {len(homographies)} homographies for video {video_id} "
        f"using {total_lines_used} line constraints total"
    )

    # Convert info keys to strings for JSON serialization
    info_serialized = {str(k): v for k, v in computation_info.items()}

    return {
        "frames": sorted(homographies.keys()),
        "info": info_serialized
    }


@app.get("/line-constraints/available-lines")
async def get_available_line_ids():
    """
    Get list of available line IDs for line annotations.

    Returns dict mapping line_id to Y value in meters.
    """
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
        }
    }


@app.post("/videos/{video_id}/map_players", response_model=List[PlayerPitchPosition])
async def map_players(video_id: str):
    """
    Map player detections to pitch coordinates using computed homographies.
    
    Returns list of PlayerPitchPosition objects with source="homography".
    """
    if video_id not in videos:
        raise HTTPException(status_code=404, detail="Video not found")
    
    # Load detections
    detections = load_detections(video_id)
    if detections is None:
        raise HTTPException(status_code=400, detail="No detections found. Run tracking first.")
    
    # Load homographies
    homographies = load_homographies(video_id)
    if homographies is None:
        raise HTTPException(status_code=400, detail="No homographies found. Compute homographies first.")
    
    # Map players to pitch
    try:
        positions = map_players_to_pitch(detections, homographies)
        player_positions_cache[video_id] = positions
    except Exception as e:
        logger.error(f"Player mapping failed for video {video_id}: {e}")
        raise HTTPException(status_code=500, detail="Player mapping failed")

    logger.info(f"Mapped {len(positions)} player positions for video {video_id}")

    return positions


@app.post("/videos/{video_id}/interpolate", response_model=InterpolationResponse)
async def interpolate_trajectories_endpoint(
    video_id: str,
    start_frame: int = Query(0, description="First frame to interpolate"),
    end_frame: int = Query(100, description="Last frame to interpolate (inclusive)")
):
    """
    Interpolate player trajectories between anchor frames.
    
    Args:
        video_id: Video identifier
        start_frame: First frame to interpolate
        end_frame: Last frame to interpolate (inclusive)
    
    Returns:
        InterpolationResponse with number of frames generated
    """
    if video_id not in videos:
        raise HTTPException(status_code=404, detail="Video not found")
    
    # Validate frame range
    if start_frame < 0 or end_frame < start_frame:
        raise HTTPException(status_code=400, detail="Invalid frame range")

    # Get sparse positions (from mapping)
    sparse_positions = player_positions_cache.get(video_id)
    if sparse_positions is None:
        raise HTTPException(
            status_code=400,
            detail="No player positions found. Run map_players first."
        )
    
    # Filter to only homography-sourced positions
    homography_positions = [
        p for p in sparse_positions if p.source == "homography"
    ]
    
    if not homography_positions:
        raise HTTPException(
            status_code=400,
            detail="No homography-based positions found for interpolation"
        )
    
    # Interpolate
    try:
        interpolated = interpolate_trajectories(
            homography_positions,
            start_frame,
            end_frame
        )
        
        # Update cache with interpolated positions
        # Remove old interpolated positions in range
        existing = player_positions_cache.get(video_id, [])
        existing_filtered = [
            p for p in existing
            if not (start_frame <= p.frame_idx <= end_frame and p.source == "interpolated")
        ]
        
        # Add new interpolated positions
        player_positions_cache[video_id] = existing_filtered + interpolated
        
        # Count newly generated frames
        frames_generated = len([
            p for p in interpolated if p.source == "interpolated"
        ])
        
    except Exception as e:
        logger.error(f"Interpolation failed for video {video_id}: {e}")
        raise HTTPException(status_code=500, detail="Interpolation failed")

    logger.info(f"Generated {frames_generated} interpolated frames for video {video_id}")

    return InterpolationResponse(
        frames_generated=frames_generated,
        method="linear"
    )


@app.get("/videos/{video_id}/players", response_model=List[PlayerPitchPosition])
async def get_player_positions(video_id: str):
    """
    Get all player positions (sparse + interpolated) for a video.
    
    Returns list of PlayerPitchPosition objects sorted by frame_idx and track_id.
    """
    if video_id not in videos:
        raise HTTPException(status_code=404, detail="Video not found")
    
    positions = player_positions_cache.get(video_id)
    if positions is None:
        raise HTTPException(
            status_code=404,
            detail="No player positions found. Run map_players first."
        )
    
    # Sort by frame_idx, then track_id
    positions_sorted = sorted(positions, key=lambda p: (p.frame_idx, p.track_id))
    
    return positions_sorted


@app.post("/process-video", response_model=ProcessVideoResponse)
async def process_video(
    file: UploadFile = File(...),
    annotations_json: str = Form(..., description="JSON string of pitch annotations for anchor frames (supports both v1 PitchAnnotation and v2 AnchorFrameAnnotation with lines)"),
    start_frame: int = Form(0, description="First frame to process (for trimming)"),
    end_frame: Optional[int] = Form(None, description="Last frame to process (for trimming), None means end of video")
):
    """
    Unified endpoint to process a video through the entire pipeline.
    
    Steps:
    1. Upload video
    2. Run YOLOv8n + ByteTrack tracking
    3. Compute homographies at anchor frames (with line constraints if provided)
    4. Map players to pitch coordinates
    5. Interpolate player trajectories
    6. Return JSON pitch coordinates
    
    Args:
        file: MP4 video file
        annotations_json: JSON string of pitch annotations for anchor frames.
                         Supports both formats:
                         - v1: [{"frame_idx": 0, "points": [...]}]
                         - v2: [{"frame_idx": 0, "points": [...], "lines": [...]}]
        start_frame: First frame to process (for trimming)
        end_frame: Last frame to process (for trimming), None means end of video

    Returns:
        ProcessVideoResponse with video_id, status, and player_positions
    """
    # Read and validate file
    content = await file.read()
    validate_video_upload(file, content)

    # Step 1: Upload video
    video_id = str(uuid.uuid4())
    video_path = VIDEOS_DIR / f"{video_id}.mp4"
    
    try:
        with open(video_path, "wb") as f:
            f.write(content)
        
        # Extract metadata
        metadata = get_video_metadata(str(video_path))
        videos[video_id] = {
            "path": str(video_path),
            "fps": metadata["fps"],
            "num_frames": metadata["num_frames"]
        }
        logger.info(f"Processing video {video_id}: {metadata['num_frames']} frames")
    except Exception as e:
        if video_path.exists():
            video_path.unlink()
        logger.error(f"Failed to upload video: {e}")
        raise HTTPException(status_code=400, detail="Failed to upload video. Ensure it is a valid MP4 file.")

    try:
        # Step 2: Run YOLOv8n + ByteTrack tracking
        from pipeline.detect import run_tracking  # Import here to avoid top-level ML imports
        logger.info(f"Running tracking on video {video_id}")
        detections = run_tracking(str(video_path))
        if not detections:
            raise HTTPException(status_code=500, detail="No detections found in video")

        detections_cache[video_id] = detections
        save_detections(video_id, detections)
        
        # Step 3: Compute homographies at anchor frames
        # Parse annotations JSON - detect format (v1 or v2)
        try:
            annotations_list = json.loads(annotations_json)
        except json.JSONDecodeError as e:
            raise HTTPException(status_code=400, detail=f"Invalid JSON in annotations: {e}")

        # Check if annotations include line constraints (v2 format)
        has_lines = any('lines' in ann and ann['lines'] for ann in annotations_list)

        if has_lines:
            # Use v2 line-constrained homography
            logger.info(f"Using line-constrained homography (v2) for video {video_id}")
            try:
                annotations = [AnchorFrameAnnotation(**ann) for ann in annotations_list]
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Invalid v2 annotations format: {e}")

            annotations_dict = {}
            for ann in annotations:
                annotations_dict[ann.frame_idx] = {
                    "keypoints": ann.points,
                    "lines": ann.lines
                }

            homographies, computation_info = compute_homographies_with_lines(
                annotations_dict,
                num_samples_per_line=10,
                max_iterations=3,
                keypoint_weight=3
            )

            # Log line constraint usage
            total_lines = sum(info.get('valid_lines', 0) for info in computation_info.values())
            logger.info(f"Line-constrained homography used {total_lines} line constraints")
        else:
            # Use v1 standard homography (backwards compatible)
            logger.info(f"Using standard homography (v1) for video {video_id}")
            try:
                annotations = [PitchAnnotation(**ann) for ann in annotations_list]
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Invalid annotations format: {e}")

            annotations_dict = {}
            for ann in annotations:
                annotations_dict[ann.frame_idx] = ann.points

            homographies = compute_homographies_from_annotations(annotations_dict)

        if not homographies:
            raise HTTPException(status_code=400, detail="No valid homographies computed. Need at least 4 points per frame.")
        
        homographies_cache[video_id] = homographies
        save_homographies(video_id, homographies)
        
        # Step 4: Map players to pitch coordinates
        positions = map_players_to_pitch(detections, homographies)
        if not positions:
            raise HTTPException(status_code=500, detail="Failed to map players to pitch")
        
        player_positions_cache[video_id] = positions
        
        # Step 5: Interpolate trajectories
        # Use provided frame range, or default to full video
        num_frames = metadata["num_frames"]
        actual_start = max(0, start_frame)
        actual_end = min(num_frames - 1, end_frame) if end_frame is not None else num_frames - 1

        # Filter positions to the requested frame range
        positions_in_range = [p for p in positions if actual_start <= p.frame_idx <= actual_end]

        interpolated = interpolate_trajectories(positions_in_range, actual_start, actual_end)

        # interpolate_trajectories already returns both original anchor positions
        # and interpolated positions, so we use it directly
        all_positions_sorted = sorted(interpolated, key=lambda p: (p.frame_idx, p.track_id))
        player_positions_cache[video_id] = all_positions_sorted
        
        # Get frame range info
        actual_start_frame = start_frame if start_frame else 0
        actual_end_frame = end_frame if end_frame else metadata.get('num_frames', 0) - 1
        video_fps = metadata.get('fps', 25)

        logger.info(f"Processing complete for video {video_id}: {len(all_positions_sorted)} positions")

        return ProcessVideoResponse(
            video_id=video_id,
            status="completed",
            player_positions=all_positions_sorted,
            homography_frames=list(homographies.keys()),
            start_frame=actual_start_frame,
            end_frame=actual_end_frame,
            fps=video_fps
        )
        
    except HTTPException:
        raise
    except Exception as e:
        # Cleanup on error
        if video_path.exists():
            video_path.unlink()
        logger.error(f"Processing failed for video {video_id}: {e}")
        raise HTTPException(status_code=500, detail="Processing failed. Please try again.")
