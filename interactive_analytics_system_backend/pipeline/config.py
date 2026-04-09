"""Configuration constants for the video analysis pipeline."""
import os

from pipeline.gaa_pitch_config import CANVAS_W as OUT_W, CANVAS_H as OUT_H  # noqa: F401

# YOLO model path from environment variable
YOLO_MODEL_PATH = os.getenv("YOLO_MODEL_PATH", "models/v8s_960_v9.pt")

# Default tracking confidence threshold
DEFAULT_CONF = 0.35