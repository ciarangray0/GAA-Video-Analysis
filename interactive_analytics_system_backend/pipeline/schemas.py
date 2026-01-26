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


class LineAnnotation(BaseModel):
    """A pitch line annotation defined by two points.

    Used to provide additional constraints for homography computation
    in regions where point intersections are not visible (e.g., midfield).

    The user clicks two points on a known horizontal pitch line, and the
    system uses the known Y-value of that line to generate synthetic
    point correspondences.

    Attributes:
        line_id: Identifier for the line (e.g., "20m_top", "halfway", "65m_bottom").
                 Must match a key in GAA_PITCH_LINES.
        u1, v1: First point on the line in image pixel coordinates.
        u2, v2: Second point on the line in image pixel coordinates.
    """
    line_id: str
    u1: float
    v1: float
    u2: float
    v2: float


class AnchorFrameAnnotation(BaseModel):
    """Complete annotation for a single anchor frame.

    Supports both traditional keypoint annotations and line annotations
    for improved homography stability.

    Attributes:
        frame_idx: Frame index in the video.
        points: List of point keypoint annotations (corners, goal posts, etc.).
        lines: Optional list of line annotations for additional constraints.
    """
    frame_idx: int
    points: List[PitchPoint]
    lines: List[LineAnnotation] = []


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
    width: int
    height: int
    duration_seconds: float


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
    homography_frames: Optional[List[int]] = None  # Frames where homographies were computed
    start_frame: Optional[int] = None  # First frame processed
    end_frame: Optional[int] = None  # Last frame processed
    fps: Optional[float] = None  # Video FPS for playback
