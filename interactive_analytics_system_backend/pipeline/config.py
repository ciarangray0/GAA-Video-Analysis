"""Configuration constants for the video analysis pipeline."""
import os

# Pitch dimensions (meters) - includes goal area extending 5m behind each goal line
PITCH_W = 85.0
PITCH_H = 145.0  # 140m pitch + 5m for goal area

# Output canvas dimensions (pixels) - 10:1 scale from meters
OUT_W = 850
OUT_H = 1450

# Radial distortion coefficient
K1 = 8e-8

# YOLO model path from environment variable
YOLO_MODEL_PATH = os.getenv("YOLO_MODEL_PATH", "models/best.pt")

# Default tracking confidence threshold
DEFAULT_CONF = 0.25
