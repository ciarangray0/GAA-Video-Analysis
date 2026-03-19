"""YOLO + BotSort detection and tracking via remote GPU (Modal)."""
from typing import List
import logging

from pipeline.schemas import Detection

logger = logging.getLogger(__name__)


def run_tracking(video_path: str) -> List[Detection]:
    """
    Run YOLO detection and BotSort tracking on a video via remote GPU.

    Args:
        video_path: Path to input video file

    Returns:
        List of Detection objects with frame_idx, track_id, bbox, confidence
    """
    from gpu_inference import get_gpu_client

    client = get_gpu_client()
    logger.info(f"Running tracking on remote GPU ({client.provider.value})")
    return client.track_video(video_path)