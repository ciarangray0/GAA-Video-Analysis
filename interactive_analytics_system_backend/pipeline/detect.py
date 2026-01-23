"""YOLO + ByteTrack detection and tracking."""
from typing import List
import numpy as np
from ultralytics import YOLO

from pipeline.schemas import Detection
from pipeline.config import YOLO_MODEL_PATH, DEFAULT_CONF


def run_tracking(
    video_path: str,
    model_path: str = None,
    conf: float = None
) -> List[Detection]:
    """
    Run YOLO detection and ByteTrack tracking on a video.
    
    Args:
        video_path: Path to input video file
        model_path: Path to YOLO model weights (defaults to YOLO_MODEL_PATH)
        conf: Confidence threshold (defaults to DEFAULT_CONF)
    
    Returns:
        List of Detection objects with frame_idx, track_id, bbox, confidence
    """
    if model_path is None:
        model_path = YOLO_MODEL_PATH
    if conf is None:
        conf = DEFAULT_CONF
    
    model = YOLO(model_path)
    
    results = model.track(
        source=video_path,
        tracker="bytetrack.yaml",
        conf=conf,
        save=False,
        stream=False
    )
    
    detections = []
    
    for frame_idx, r in enumerate(results):
        if r.boxes is None:
            continue
        
        boxes = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        ids = r.boxes.id.cpu().numpy()
        
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
