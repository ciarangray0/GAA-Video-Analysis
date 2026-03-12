"""In-memory store for video analysis state."""
from typing import Dict, List, Any

from pipeline.schemas import Detection, PlayerPitchPosition


class VideoStore:
    """Isolated container for all module-level in-memory state."""

    def __init__(self):
        self.videos: Dict[str, dict] = {}
        self.detections_cache: Dict[str, List[Detection]] = {}
        self.homographies_cache: Dict[str, Dict[int, Any]] = {}
        self.anchor_homographies_cache: Dict[str, Dict[int, Any]] = {}
        self.v3_anchor_homographies_cache: Dict[str, Dict[int, Any]] = {}
        self.v3_homographies_cache: Dict[str, Dict[int, Any]] = {}
        self.player_positions_cache: Dict[str, List[PlayerPitchPosition]] = {}


store = VideoStore()
