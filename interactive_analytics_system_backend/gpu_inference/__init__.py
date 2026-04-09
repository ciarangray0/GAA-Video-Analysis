"""
GPU Inference Client

This module provides a client to call the remote GPU inference service
(Modal, RunPod, or other) for YOLO + ByteTrack tracking.
"""

import httpx
import base64
import logging
from typing import List, Optional
from enum import Enum
import os

logger = logging.getLogger(__name__)


class GPUProvider(str, Enum):
    """Supported GPU inference providers."""
    MODAL = "modal"


class GPUInferenceClient:
    """Client for remote GPU inference services."""

    def __init__(
        self,
        provider: GPUProvider = GPUProvider.MODAL,
        endpoint_url: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: float = 600.0,  # 10 minute timeout for long videos
    ):
        """
        Initialize GPU inference client.

        Args:
            provider: Which GPU provider to use
            endpoint_url: The endpoint URL for the GPU service
            api_key: API key for authentication (if required)
            timeout: Request timeout in seconds
        """
        self.provider = provider
        self.endpoint_url = endpoint_url
        self.api_key = api_key
        self.timeout = timeout

        self._client = httpx.Client(timeout=timeout)

    def track_video(self, video_path: str) -> List:
        """
        Run YOLO + BotSort tracking on a video using remote GPU.

        Args:
            video_path: Path to the video file

        Returns:
            List of Detection objects
        """
        # Import here to avoid circular imports
        from pipeline.schemas import Detection

        if self.provider == GPUProvider.MODAL:
            return self._track_modal(video_path)
        else:
            raise ValueError(f"Unknown provider: {self.provider}")

    def _track_modal(self, video_path: str) -> List:
        """Track using Modal GPU endpoint."""
        from pipeline.schemas import Detection

        if not self.endpoint_url:
            raise ValueError(
                "GPU_ENDPOINT_URL is required for Modal provider. "
                "Deploy modal_yolo.py and set the endpoint URL."
            )

        # Read and encode video
        with open(video_path, "rb") as f:
            video_bytes = f.read()

        video_base64 = base64.b64encode(video_bytes).decode("utf-8")

        logger.info(f"Sending {len(video_bytes) / 1024 / 1024:.2f} MB to Modal GPU...")

        # Call Modal endpoint
        response = self._client.post(
            self.endpoint_url,
            json={"video_base64": video_base64},
            headers={"Content-Type": "application/json"},
        )
        response.raise_for_status()

        data = response.json()

        if data.get("status") == "error":
            raise RuntimeError(f"Modal tracking failed: {data.get('message')}")

        # Convert to Detection objects
        detections = []
        for det in data.get("detections", []):
            detections.append(Detection(
                frame_idx=det["frame_idx"],
                track_id=det["track_id"],
                x1=det["x1"],
                y1=det["y1"],
                x2=det["x2"],
                y2=det["y2"],
                confidence=det["confidence"],
                class_name=det.get("class_name", "GAA-player-lablers"),
            ))

        logger.info(f"Received {len(detections)} detections from Modal GPU")
        return detections

    def close(self):
        """Close the HTTP client."""
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


# Singleton instance for the app
_gpu_client: Optional[GPUInferenceClient] = None


def get_gpu_client() -> GPUInferenceClient:
    """Get or create the GPU inference client singleton."""
    global _gpu_client

    if _gpu_client is None:
        provider = os.getenv("GPU_PROVIDER", "local")
        endpoint_url = os.getenv("GPU_ENDPOINT_URL")
        api_key = os.getenv("GPU_API_KEY")

        # Sanitize endpoint URL - strip whitespace and newlines
        if endpoint_url:
            endpoint_url = endpoint_url.strip().replace('\n', '').replace('\r', '')

        _gpu_client = GPUInferenceClient(
            provider=GPUProvider(provider),
            endpoint_url=endpoint_url,
            api_key=api_key,
        )

        logger.info(f"GPU inference client initialized with provider: {provider}")

    return _gpu_client
