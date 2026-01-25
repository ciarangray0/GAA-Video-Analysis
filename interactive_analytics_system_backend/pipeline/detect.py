"""YOLO + ByteTrack detection and tracking.

This module provides tracking functionality that can run either:
1. On a remote GPU via Modal (recommended for production)
2. Locally on CPU (slow, for testing only)

The GPU_PROVIDER environment variable controls which is used.
"""
from typing import List
import os
import logging

from pipeline.schemas import Detection

logger = logging.getLogger(__name__)


def run_tracking(video_path: str) -> List[Detection]:
    """
    Run YOLO detection and ByteTrack tracking on a video.
    
    This function delegates to either:
    - Remote GPU inference (Modal) if GPU_PROVIDER is set
    - Local CPU inference as fallback

    Args:
        video_path: Path to input video file

    Returns:
        List of Detection objects with frame_idx, track_id, bbox, confidence
    """
    provider = os.getenv("GPU_PROVIDER", "local")

    if provider != "local":
        # Use remote GPU inference
        return _run_tracking_remote(video_path)
    else:
        # Fall back to local CPU
        logger.warning("Using local CPU tracking - this will be slow!")
        return _run_tracking_local(video_path)


def _run_tracking_remote(video_path: str) -> List[Detection]:
    """Run tracking on remote GPU via Modal/RunPod."""
    from gpu_inference import get_gpu_client

    client = get_gpu_client()
    logger.info(f"Running tracking on remote GPU ({client.provider.value})")

    return client.track_video(video_path)


def _run_tracking_local(video_path: str) -> List[Detection]:
    """
    Run YOLO detection and ByteTrack tracking locally on CPU.

    Note: This requires ultralytics and torch to be installed.
    These are commented out in requirements.txt by default.
    """
    from pipeline.config import YOLO_MODEL_PATH, DEFAULT_CONF

    # Import YOLO lazily to avoid heavy dependency at module import time
    try:
        from ultralytics import YOLO
    except ImportError as e:
        raise RuntimeError(
            "ultralytics is not installed. Either:\n"
            "1. Set GPU_PROVIDER=modal and configure GPU_ENDPOINT_URL, or\n"
            "2. Uncomment torch/ultralytics in requirements.txt for local CPU fallback"
        ) from e

    model = YOLO(YOLO_MODEL_PATH)

    # Force CPU usage
    results = model.track(
        source=video_path,
        tracker="bytetrack.yaml",
        conf=DEFAULT_CONF,
        save=False,
        stream=False,
        device="cpu"
    )
    
    detections = []
    
    for frame_idx, r in enumerate(results):
        if r.boxes is None:
            continue
        
        boxes = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        ids = r.boxes.id

        # Skip frames where tracking failed (no IDs)
        if ids is None:
            continue

        ids = ids.cpu().numpy()

        for box, conf_score, tid in zip(boxes, confs, ids):
            x1, y1, x2, y2 = box
            detections.append(Detection(
                frame_idx=frame_idx,
                track_id=int(tid),
                x1=float(x1),
                y1=float(y1),
                x2=float(x2),
                y2=float(y2),
                confidence=float(conf_score)
            ))
    
    return detections
