"""Video utilities for metadata extraction."""
import cv2
from pathlib import Path


def get_video_metadata(video_path: str) -> dict:
    """
    Extract video metadata (fps, num_frames, width, height).
    
    Args:
        video_path: Path to video file
    
    Returns:
        Dict with 'fps', 'num_frames', 'width', 'height'
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Failed to open video: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    cap.release()
    
    return {
        'fps': int(fps) if fps > 0 else 30,
        'num_frames': num_frames,
        'width': width,
        'height': height
    }
