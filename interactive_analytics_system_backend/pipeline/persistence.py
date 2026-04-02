"""Disk persistence helpers for the GAA Video Analysis pipeline.

All JSON and file I/O is centralised here so that app.py contains no
direct open() / json.dump() calls.
"""
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).parent.parent.parent / ".env")

from pipeline.schemas import Detection

# ---------------------------------------------------------------------------
# Directory layout
# ---------------------------------------------------------------------------

DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
VIDEOS_DIR = DATA_DIR / "videos"
TRACKS_DIR = DATA_DIR / "tracks"
ANNOTATIONS_DIR = DATA_DIR / "annotations"


def ensure_dirs() -> None:
    """Create data directories if they do not exist (called on startup)."""
    VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
    TRACKS_DIR.mkdir(parents=True, exist_ok=True)
    ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Low-level JSON helpers
# ---------------------------------------------------------------------------

def _save_json(path: Path, data: Any) -> None:
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def _load_json(path: Path) -> Any:
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


# ---------------------------------------------------------------------------
# Homography serialisation helpers
# ---------------------------------------------------------------------------

def _serialize_H(h_dict: Dict[int, np.ndarray]) -> dict:
    return {str(k): v.tolist() for k, v in h_dict.items()}


def _deserialize_H(data: dict) -> Dict[int, np.ndarray]:
    return {int(k): np.array(v) for k, v in data.items()}


# ---------------------------------------------------------------------------
# Video metadata
# ---------------------------------------------------------------------------

def save_video_meta(video_id: str, meta: dict) -> None:
    _save_json(VIDEOS_DIR / f"{video_id}_meta.json", {"video_id": video_id, **meta})


def save_video_file(video_id: str, content: bytes) -> Path:
    """Write raw MP4 bytes to disk and return the saved path."""
    video_path = VIDEOS_DIR / f"{video_id}.mp4"
    with open(video_path, "wb") as f:
        f.write(content)
    return video_path


# ---------------------------------------------------------------------------
# Detections (BotSort tracks)
# ---------------------------------------------------------------------------

def save_detections(video_id: str, detections: List[Detection]) -> None:
    _save_json(TRACKS_DIR / f"{video_id}.json", [d.model_dump() for d in detections])


def load_detections(video_id: str) -> Optional[List[Detection]]:
    data = _load_json(TRACKS_DIR / f"{video_id}.json")
    return [Detection(**d) for d in data] if data is not None else None


# ---------------------------------------------------------------------------
# Homographies
# ---------------------------------------------------------------------------

def save_homography_dict(video_id: str, key: str, h_dict: Dict[int, np.ndarray]) -> None:
    _save_json(ANNOTATIONS_DIR / f"{video_id}_{key}.json", _serialize_H(h_dict))


def load_homography_dict(video_id: str, key: str) -> Optional[Dict[int, np.ndarray]]:
    data = _load_json(ANNOTATIONS_DIR / f"{video_id}_{key}.json")
    return _deserialize_H(data) if data is not None else None


# ---------------------------------------------------------------------------
# Annotations
# ---------------------------------------------------------------------------

def _serialise_ann_value(obj: Any) -> Any:
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


# ---------------------------------------------------------------------------
# Team classifications
# ---------------------------------------------------------------------------

def save_team_classifications(video_id: str, classifications: Dict[int, dict]) -> None:
    _save_json(
        ANNOTATIONS_DIR / f"{video_id}_team_classifications.json",
        {str(k): v for k, v in classifications.items()},
    )


def load_team_classifications(video_id: str) -> Optional[Dict[int, dict]]:
    data = _load_json(ANNOTATIONS_DIR / f"{video_id}_team_classifications.json")
    return {int(k): v for k, v in data.items()} if data is not None else None


# ---------------------------------------------------------------------------
# Startup restore
# ---------------------------------------------------------------------------

def restore_videos_from_disk() -> Dict[str, dict]:
    """Return a dict of video_id → meta for all saved videos on disk."""
    videos: Dict[str, dict] = {}
    for meta_path in VIDEOS_DIR.glob("*_meta.json"):
        meta = _load_json(meta_path)
        if meta is None:
            continue
        video_id = meta.get("video_id")
        video_path = Path(meta.get("path", ""))
        if not video_id or not video_path.exists():
            continue
        videos[video_id] = {k: v for k, v in meta.items() if k != "video_id"}
    return videos