"""Video utilities for metadata extraction and frame extraction."""
import cv2
from typing import Optional


def get_video_metadata(video_path: str) -> dict:
    """
    Extract video metadata (fps, num_frames, width, height, duration).

    Args:
        video_path: Path to video file
    
    Returns:
        Dict with 'fps', 'num_frames', 'width', 'height', 'duration_seconds'
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Failed to open video: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    cap.release()
    
    fps_val = fps if fps > 0 else 30
    duration_seconds = num_frames / fps_val if fps_val > 0 else 0

    return {
        'fps': int(fps_val),
        'num_frames': num_frames,
        'width': width,
        'height': height,
        'duration_seconds': round(duration_seconds, 2)
    }


def extract_frame(video_path: str, frame_idx: int) -> Optional[bytes]:
    """
    Extract a single frame from a video and return as JPEG bytes.

    Args:
        video_path: Path to video file
        frame_idx: Frame index to extract (0-based)

    Returns:
        JPEG bytes of the frame, or None if extraction fails
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"Failed to open video: {video_path}")

    # Seek to frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        return None

    # Encode as JPEG
    success, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])

    if not success:
        return None

    return buffer.tobytes()


