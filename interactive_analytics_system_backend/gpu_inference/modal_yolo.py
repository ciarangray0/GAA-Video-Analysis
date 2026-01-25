"""
Modal GPU inference service for YOLO + ByteTrack tracking.

This runs on Modal's serverless GPUs and exposes a web endpoint
that your Render backend can call.

Setup:
1. pip install modal
2. modal token new  (authenticate with Modal)
3. modal deploy modal_yolo.py  (deploy the service)

The deployed endpoint URL will be printed - use that in your backend.
"""

import modal
from typing import List, Dict, Any
import json

# Define the Modal app
app = modal.App("gaa-yolo-tracking")

# Define the container image with all dependencies
yolo_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "ultralytics>=8.0.0",
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "opencv-python-headless>=4.8.0",
        "numpy>=1.24.0",
        "lap>=0.4.0",  # Required for ByteTrack
        "fastapi[standard]",  # Required for web endpoints
    )
)

# Volume to cache the YOLO model weights
model_cache = modal.Volume.from_name("yolo-model-cache", create_if_missing=True)


@app.cls(
    image=yolo_image,
    gpu="T4",  # Use T4 GPU (cheapest, good for inference)
    timeout=600,  # 10 minute timeout
    volumes={"/model_cache": model_cache},
    scaledown_window=60,  # Keep warm for 60 seconds (renamed from container_idle_timeout)
)
class YOLOTracker:
    """YOLO + ByteTrack tracker running on GPU."""

    @modal.enter()
    def load_model(self):
        """Load YOLO model when container starts."""
        from ultralytics import YOLO
        import os

        # Set cache directory
        os.environ["YOLO_CONFIG_DIR"] = "/model_cache"

        # Load YOLOv8n model (will download on first run, then cached)
        self.model = YOLO("yolov8n.pt")
        print("YOLO model loaded successfully")

    @modal.method()
    def track_video(self, video_bytes: bytes) -> List[Dict[str, Any]]:
        """
        Run YOLO + ByteTrack on video bytes.

        Args:
            video_bytes: Raw video file bytes

        Returns:
            List of detections with frame_idx, track_id, bbox, confidence
        """
        import tempfile
        import os

        # Save video to temp file
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(video_bytes)
            temp_path = f.name

        try:
            # Run tracking
            results = self.model.track(
                source=temp_path,
                tracker="bytetrack.yaml",
                persist=True,
                conf=0.3,
                iou=0.5,
                classes=[0],  # Only detect people (class 0 in COCO)
                verbose=False,
                stream=True,  # Stream results to reduce memory
            )

            detections = []
            for frame_idx, result in enumerate(results):
                if result.boxes is None or len(result.boxes) == 0:
                    continue

                boxes = result.boxes

                # Get track IDs (may be None if tracking fails)
                track_ids = boxes.id
                if track_ids is None:
                    continue

                track_ids = track_ids.cpu().numpy().astype(int)
                bboxes = boxes.xyxy.cpu().numpy()
                confidences = boxes.conf.cpu().numpy()

                for i, track_id in enumerate(track_ids):
                    x1, y1, x2, y2 = bboxes[i]
                    detections.append({
                        "frame_idx": frame_idx,
                        "track_id": int(track_id),
                        "x1": float(x1),
                        "y1": float(y1),
                        "x2": float(x2),
                        "y2": float(y2),
                        "confidence": float(confidences[i]),
                    })

            print(f"Tracking complete: {len(detections)} detections across {frame_idx + 1} frames")
            return detections

        finally:
            # Clean up temp file
            os.unlink(temp_path)

    @modal.fastapi_endpoint(method="POST")
    def track_video_endpoint(self, video_base64: str) -> Dict[str, Any]:
        """
        Web endpoint for tracking video.

        Accepts base64-encoded video, returns JSON detections.
        """
        import base64

        try:
            video_bytes = base64.b64decode(video_base64)
            detections = self.track_video(video_bytes)
            return {
                "status": "success",
                "detections": detections,
                "count": len(detections),
            }
        except Exception as e:
            return {
                "status": "error",
                "message": str(e),
            }


# For direct function calls (not via web endpoint)
@app.function(
    image=yolo_image,
    gpu="T4",
    timeout=600,
    volumes={"/model_cache": model_cache},
)
def track_video_direct(video_bytes: bytes) -> List[Dict[str, Any]]:
    """Direct function call for tracking (for Modal SDK calls)."""
    tracker = YOLOTracker()
    return tracker.track_video(video_bytes)


# Local entrypoint for testing
@app.local_entrypoint()
def main():
    """Test the tracking with a sample video."""
    import sys

    if len(sys.argv) < 2:
        print("Usage: modal run modal_yolo.py -- <video_path>")
        return

    video_path = sys.argv[1]
    with open(video_path, "rb") as f:
        video_bytes = f.read()

    print(f"Sending {len(video_bytes)} bytes to Modal...")
    detections = track_video_direct.remote(video_bytes)
    print(f"Got {len(detections)} detections")
    print(json.dumps(detections[:5], indent=2))  # Print first 5

