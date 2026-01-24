"""Pydantic schemas for API requests and responses."""
from pydantic import BaseModel
from typing import List, Optional


class PitchPoint(BaseModel):
    """A single pitch annotation point."""
    pitch_id: str
    x_img: float
    y_img: float


class PitchAnnotation(BaseModel):
    """Pitch annotation for a single frame."""
    frame_idx: int
    points: List[PitchPoint]


class Detection(BaseModel):
    """Player detection from YOLO + ByteTrack."""
    frame_idx: int
    track_id: int
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float


class PlayerPitchPosition(BaseModel):
    """Player position mapped to pitch coordinates."""
    frame_idx: int
    track_id: int
    x_pitch: float
    y_pitch: float
    source: str  # "homography" or "interpolated"


class VideoCreateResponse(BaseModel):
    """Response after uploading a video."""
    video_id: str
    fps: int
    num_frames: int


class TrackResponse(BaseModel):
    """Response after running tracking."""
    frames_processed: int
    tracks: int


class HomographyResponse(BaseModel):
    """Response after computing homographies."""
    frames: List[int]


class InterpolationResponse(BaseModel):
    """Response after interpolating trajectories."""
    frames_generated: int
    method: str


class ProcessVideoResponse(BaseModel):
    """Response from unified /process-video endpoint."""
    video_id: str
    status: str  # "processing" or "completed"
    player_positions: Optional[List[PlayerPitchPosition]] = None
