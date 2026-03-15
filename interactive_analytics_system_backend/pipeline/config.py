"""Configuration constants for the video analysis pipeline."""
import os

# Output canvas dimensions (pixels) - 10:1 scale from meters
OUT_W = 850
OUT_H = 1400

# YOLO model path from environment variable
YOLO_MODEL_PATH = os.getenv("YOLO_MODEL_PATH", "models/v8s_960_v9.pt")

# Default tracking confidence threshold
DEFAULT_CONF = 0.35
