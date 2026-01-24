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

from pipeline.schemas import (
    VideoCreateResponse,
    PitchAnnotation,
    TrackResponse,
    HomographyResponse,
    InterpolationResponse,
    Detection,
    PlayerPitchPosition,
    ProcessVideoResponse
)
# NOTE: `run_tracking` performs heavy ML imports; import lazily inside endpoints to keep module import lightweight.
# from pipeline.detect import run_tracking
from pipeline.homography import compute_homographies_from_annotations
from pipeline.map_players import map_players_to_pitch
from pipeline.trajectories import interpolate_trajectories
from pipeline.video import get_video_metadata

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
        "num_frames": metadata["num_frames"]
    }
    
    logger.info(f"Uploaded video {video_id}: {metadata['num_frames']} frames at {metadata['fps']} fps")

    return VideoCreateResponse(
        video_id=video_id,
        fps=metadata["fps"],
        num_frames=metadata["num_frames"]
    )


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
        # Run tracking
        try:
            from pipeline.detect import run_tracking  # Import here to avoid top-level ML imports
            logger.info(f"Running tracking on video {video_id}")
            detections = run_tracking(video_path)
            detections_cache[video_id] = detections
            save_detections(video_id, detections)
        except Exception as e:
            logger.error(f"Tracking failed for video {video_id}: {e}")
            raise HTTPException(status_code=500, detail="Tracking failed. Please try again.")
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
    annotations_json: str = Form(..., description="JSON string of pitch annotations for anchor frames")
):
    """
    Unified endpoint to process a video through the entire pipeline.
    
    Steps:
    1. Upload video
    2. Run YOLOv8n + ByteTrack tracking
    3. Compute homographies at anchor frames
    4. Map players to pitch coordinates
    5. Interpolate player trajectories
    6. Return JSON pitch coordinates
    
    Args:
        file: MP4 video file
        annotations_json: JSON string of pitch annotations for anchor frames

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
        # Parse annotations JSON
        try:
            annotations_list = json.loads(annotations_json)
            annotations = [PitchAnnotation(**ann) for ann in annotations_list]
        except Exception as e:
            raise HTTPException(status_code=400, detail="Invalid annotations format. Expected JSON array of PitchAnnotation objects.")

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
        # Get frame range from video metadata
        num_frames = metadata["num_frames"]
        interpolated = interpolate_trajectories(positions, 0, num_frames - 1)
        
        # Combine sparse and interpolated positions
        all_positions = positions + [
            p for p in interpolated if p.source == "interpolated"
        ]
        
        # Sort by frame_idx, then track_id
        all_positions_sorted = sorted(all_positions, key=lambda p: (p.frame_idx, p.track_id))
        player_positions_cache[video_id] = all_positions_sorted
        
        logger.info(f"Processing complete for video {video_id}: {len(all_positions_sorted)} positions")

        return ProcessVideoResponse(
            video_id=video_id,
            status="completed",
            player_positions=all_positions_sorted
        )
        
    except HTTPException:
        raise
    except Exception as e:
        # Cleanup on error
        if video_path.exists():
            video_path.unlink()
        logger.error(f"Processing failed for video {video_id}: {e}")
        raise HTTPException(status_code=500, detail="Processing failed. Please try again.")
