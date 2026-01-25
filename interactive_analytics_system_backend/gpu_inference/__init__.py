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
    RUNPOD = "runpod"
    LOCAL = "local"  # Fallback to local CPU


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
        Run YOLO + ByteTrack tracking on a video using remote GPU.

        Args:
            video_path: Path to the video file

        Returns:
            List of Detection objects
        """
        # Import here to avoid circular imports
        from pipeline.schemas import Detection

        if self.provider == GPUProvider.LOCAL:
            raise ValueError("Local provider should not use GPUInferenceClient")
        elif self.provider == GPUProvider.MODAL:
            return self._track_modal(video_path)
        elif self.provider == GPUProvider.RUNPOD:
            return self._track_runpod(video_path)
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
            ))

        logger.info(f"Received {len(detections)} detections from Modal GPU")
        return detections

    def _track_runpod(self, video_path: str) -> List:
        """Track using RunPod serverless endpoint."""
        from pipeline.schemas import Detection

        if not self.endpoint_url or not self.api_key:
            raise ValueError("endpoint_url and api_key are required for RunPod")

        # Read and encode video
        with open(video_path, "rb") as f:
            video_bytes = f.read()

        video_base64 = base64.b64encode(video_bytes).decode("utf-8")

        logger.info(f"Sending {len(video_bytes) / 1024 / 1024:.2f} MB to RunPod GPU...")

        # Call RunPod endpoint
        response = self._client.post(
            self.endpoint_url,
            json={"input": {"video_base64": video_base64}},
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
        )
        response.raise_for_status()

        data = response.json()

        # RunPod returns async job - poll for result
        if "id" in data:
            return self._poll_runpod_job(data["id"])

        return self._parse_runpod_response(data)

    def _poll_runpod_job(self, job_id: str) -> List:
        """Poll RunPod job until complete."""
        import time
        from pipeline.schemas import Detection

        # Extract endpoint ID from URL
        endpoint_id = self.endpoint_url.rstrip('/').split('/')[-1]
        if endpoint_id == 'run':
            endpoint_id = self.endpoint_url.rstrip('/').split('/')[-2]

        status_url = f"https://api.runpod.ai/v2/{endpoint_id}/status/{job_id}"

        while True:
            response = self._client.get(
                status_url,
                headers={"Authorization": f"Bearer {self.api_key}"},
            )
            response.raise_for_status()
            data = response.json()

            status = data.get("status")
            if status == "COMPLETED":
                return self._parse_runpod_response(data)
            elif status == "FAILED":
                raise RuntimeError(f"RunPod job failed: {data.get('error')}")

            logger.info(f"RunPod job {job_id} status: {status}")
            time.sleep(2)

    def _parse_runpod_response(self, data: dict) -> List:
        """Parse RunPod response to Detection objects."""
        from pipeline.schemas import Detection

        output = data.get("output", {})
        detections = []

        for det in output.get("detections", []):
            detections.append(Detection(
                frame_idx=det["frame_idx"],
                track_id=det["track_id"],
                x1=det["x1"],
                y1=det["y1"],
                x2=det["x2"],
                y2=det["y2"],
                confidence=det["confidence"],
            ))

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

        if provider == "local":
            raise ValueError(
                "GPU_PROVIDER is 'local'. Use run_tracking() from detect.py instead."
            )

        _gpu_client = GPUInferenceClient(
            provider=GPUProvider(provider),
            endpoint_url=endpoint_url,
            api_key=api_key,
        )

        logger.info(f"GPU inference client initialized with provider: {provider}")

    return _gpu_client
