"""Configuration constants for the video analysis pipeline."""
import os

# Pitch dimensions (meters)
PITCH_W = 85.0
PITCH_H = 145.0

# Output canvas dimensions (pixels)
OUT_W = 850
OUT_H = 1450

# Radial distortion coefficient
K1 = 8e-8

# YOLO model path from environment variable
YOLO_MODEL_PATH = os.getenv("YOLO_MODEL_PATH", "models/best.pt")

# Default tracking confidence threshold
DEFAULT_CONF = 0.25
