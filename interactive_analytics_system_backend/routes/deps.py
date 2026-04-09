"""Shared FastAPI dependencies used across route modules."""
from fastapi import HTTPException

from store import store


def get_video_or_404(video_id: str) -> dict:
    """Return the video metadata dict or raise 404."""
    if video_id not in store.videos:
        raise HTTPException(status_code=404, detail="Video not found")
    return store.videos[video_id]
