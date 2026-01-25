"""Tests for GPU inference client and detect module."""
import pytest
from unittest.mock import Mock, patch


class TestGPUInferenceClient:
    """Tests for the GPUInferenceClient class."""

    def test_gpu_provider_enum(self):
        """Test GPUProvider enum values."""
        from gpu_inference import GPUProvider

        assert GPUProvider.MODAL.value == "modal"
        assert GPUProvider.RUNPOD.value == "runpod"
        assert GPUProvider.LOCAL.value == "local"

    def test_client_initialization(self):
        """Test GPUInferenceClient initialization."""
        from gpu_inference import GPUInferenceClient, GPUProvider

        gpu_client = GPUInferenceClient(
            provider=GPUProvider.MODAL,
            endpoint_url="https://example.modal.run",
            timeout=300.0
        )

        assert gpu_client.provider == GPUProvider.MODAL
        assert gpu_client.endpoint_url == "https://example.modal.run"
        assert gpu_client.timeout == 300.0
        gpu_client.close()

    def test_client_raises_for_local_provider(self):
        """Test that GPUInferenceClient raises error for local provider."""
        from gpu_inference import GPUInferenceClient, GPUProvider

        gpu_client = GPUInferenceClient(provider=GPUProvider.LOCAL)

        with pytest.raises(ValueError, match="Local provider should not use"):
            gpu_client.track_video("/fake/path.mp4")

        gpu_client.close()

    def test_client_raises_without_endpoint_url(self, tmp_path):
        """Test that Modal tracking raises without endpoint URL."""
        from gpu_inference import GPUInferenceClient, GPUProvider

        # Create a fake video file
        video_file = tmp_path / "test.mp4"
        video_file.write_bytes(b"fake video data")

        gpu_client = GPUInferenceClient(
            provider=GPUProvider.MODAL,
            endpoint_url=None
        )

        with pytest.raises(ValueError, match="GPU_ENDPOINT_URL is required"):
            gpu_client.track_video(str(video_file))

        gpu_client.close()

    def test_modal_tracking_success(self, tmp_path):
        """Test successful Modal tracking."""
        from gpu_inference import GPUInferenceClient, GPUProvider

        # Create a fake video file
        video_file = tmp_path / "test.mp4"
        video_file.write_bytes(b"fake video data")

        # Mock response data
        mock_response_data = {
            "status": "success",
            "detections": [
                {
                    "frame_idx": 0,
                    "track_id": 1,
                    "x1": 100.0,
                    "y1": 100.0,
                    "x2": 200.0,
                    "y2": 300.0,
                    "confidence": 0.95
                },
                {
                    "frame_idx": 1,
                    "track_id": 1,
                    "x1": 105.0,
                    "y1": 102.0,
                    "x2": 205.0,
                    "y2": 302.0,
                    "confidence": 0.93
                }
            ]
        }

        gpu_client = GPUInferenceClient(
            provider=GPUProvider.MODAL,
            endpoint_url="https://test.modal.run"
        )

        # Mock the HTTP client
        mock_response = Mock()
        mock_response.json.return_value = mock_response_data
        mock_response.raise_for_status = Mock()
        gpu_client._client.post = Mock(return_value=mock_response)

        detections = gpu_client.track_video(str(video_file))

        assert len(detections) == 2
        assert detections[0].frame_idx == 0
        assert detections[0].track_id == 1
        assert detections[0].confidence == 0.95
        assert detections[1].frame_idx == 1

        # Verify the request was made correctly
        gpu_client._client.post.assert_called_once()
        call_args = gpu_client._client.post.call_args
        assert call_args[0][0] == "https://test.modal.run"
        assert "video_base64" in call_args[1]["json"]

        gpu_client.close()

    def test_modal_tracking_error_response(self, tmp_path):
        """Test Modal tracking handles error response."""
        from gpu_inference import GPUInferenceClient, GPUProvider

        video_file = tmp_path / "test.mp4"
        video_file.write_bytes(b"fake video data")

        mock_response_data = {
            "status": "error",
            "message": "Video processing failed"
        }

        gpu_client = GPUInferenceClient(
            provider=GPUProvider.MODAL,
            endpoint_url="https://test.modal.run"
        )

        mock_response = Mock()
        mock_response.json.return_value = mock_response_data
        mock_response.raise_for_status = Mock()
        gpu_client._client.post = Mock(return_value=mock_response)

        with pytest.raises(RuntimeError, match="Modal tracking failed"):
            gpu_client.track_video(str(video_file))

        gpu_client.close()

    def test_endpoint_url_sanitization(self, monkeypatch):
        """Test that endpoint URL is sanitized (newlines stripped)."""
        import gpu_inference

        # Reset singleton
        gpu_inference._gpu_client = None

        # Set env vars with newline in URL
        monkeypatch.setenv("GPU_PROVIDER", "modal")
        monkeypatch.setenv("GPU_ENDPOINT_URL", "https://test.modal.run\n")

        from gpu_inference import get_gpu_client
        gpu_client = get_gpu_client()

        assert gpu_client.endpoint_url == "https://test.modal.run"
        assert "\n" not in gpu_client.endpoint_url

        # Reset singleton for other tests
        gpu_inference._gpu_client = None

    def test_context_manager(self):
        """Test GPUInferenceClient as context manager."""
        from gpu_inference import GPUInferenceClient, GPUProvider

        with GPUInferenceClient(provider=GPUProvider.MODAL) as gpu_client:
            assert gpu_client is not None

        # Client should be closed after exiting context


class TestDetectModule:
    """Tests for the detect module."""

    def test_run_tracking_uses_gpu_when_provider_set(self, monkeypatch, tmp_path):
        """Test that run_tracking uses GPU client when GPU_PROVIDER is set."""
        # Create fake video
        video_file = tmp_path / "test.mp4"
        video_file.write_bytes(b"fake video data")

        monkeypatch.setenv("GPU_PROVIDER", "modal")
        monkeypatch.setenv("GPU_ENDPOINT_URL", "https://test.modal.run")

        # Import Detection here to avoid import issues
        from pipeline.schemas import Detection

        # Mock the GPU client
        mock_detections = [
            Detection(frame_idx=0, track_id=1, x1=100, y1=100, x2=200, y2=200, confidence=0.9)
        ]

        mock_client = Mock()
        mock_client.track_video.return_value = mock_detections
        mock_client.provider = Mock()
        mock_client.provider.value = "modal"

        # Reset the singleton
        import gpu_inference
        gpu_inference._gpu_client = None

        with patch("gpu_inference.get_gpu_client", return_value=mock_client):
            from pipeline.detect import run_tracking
            detections = run_tracking(str(video_file))

        assert len(detections) == 1
        assert detections[0].track_id == 1

    def test_run_tracking_local_fallback(self, monkeypatch, tmp_path):
        """Test that run_tracking falls back to local when GPU_PROVIDER=local."""
        monkeypatch.setenv("GPU_PROVIDER", "local")

        # Create fake video
        video_file = tmp_path / "test.mp4"
        video_file.write_bytes(b"fake video data")

        # Import Detection here
        from pipeline.schemas import Detection

        # Mock the local tracking function
        mock_detections = [
            Detection(frame_idx=0, track_id=1, x1=100, y1=100, x2=200, y2=200, confidence=0.9)
        ]

        with patch("pipeline.detect._run_tracking_local", return_value=mock_detections) as mock_local:
            from pipeline.detect import run_tracking
            detections = run_tracking(str(video_file))

        mock_local.assert_called_once()
        assert len(detections) == 1


class TestGetGpuClient:
    """Tests for get_gpu_client function."""

    def test_raises_for_local_provider(self, monkeypatch):
        """Test that get_gpu_client raises when provider is local."""
        import gpu_inference
        gpu_inference._gpu_client = None

        monkeypatch.setenv("GPU_PROVIDER", "local")

        from gpu_inference import get_gpu_client

        with pytest.raises(ValueError, match="GPU_PROVIDER is 'local'"):
            get_gpu_client()

    def test_returns_singleton(self, monkeypatch):
        """Test that get_gpu_client returns singleton instance."""
        import gpu_inference
        gpu_inference._gpu_client = None

        monkeypatch.setenv("GPU_PROVIDER", "modal")
        monkeypatch.setenv("GPU_ENDPOINT_URL", "https://test.modal.run")

        from gpu_inference import get_gpu_client

        client1 = get_gpu_client()
        client2 = get_gpu_client()

        assert client1 is client2

        # Reset for other tests
        gpu_inference._gpu_client = None

    def test_sanitizes_url_whitespace(self, monkeypatch):
        """Test that URL whitespace is stripped."""
        import gpu_inference
        gpu_inference._gpu_client = None

        monkeypatch.setenv("GPU_PROVIDER", "modal")
        monkeypatch.setenv("GPU_ENDPOINT_URL", "  https://test.modal.run  ")

        from gpu_inference import get_gpu_client

        gpu_client = get_gpu_client()
        assert gpu_client.endpoint_url == "https://test.modal.run"

        gpu_inference._gpu_client = None

