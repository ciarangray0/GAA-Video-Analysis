from pathlib import Path
import numpy as np
import pytest
from fastapi.testclient import TestClient

# Ensure project root is on sys.path so `pipeline` imports work when tests run
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Provide a lightweight fake `ultralytics` module during tests to avoid requiring heavy ML deps
import types
if "ultralytics" not in sys.modules:
    ultralytics_stub = types.ModuleType("ultralytics")
    class YOLO:
        def __init__(self, *args, **kwargs):
            pass
        def track(self, *args, **kwargs):
            return []
    ultralytics_stub.YOLO = YOLO
    sys.modules["ultralytics"] = ultralytics_stub

from pipeline.schemas import Detection, PitchAnnotation, PitchPoint, PlayerPitchPosition
from app import app, VIDEOS_DIR, TRACKS_DIR, ANNOTATIONS_DIR


@pytest.fixture(autouse=True)
def use_temp_dirs(tmp_path, monkeypatch):
    """Redirect data directories to a temporary path for tests."""
    temp_data = tmp_path / "data"
    temp_videos = temp_data / "videos"
    temp_tracks = temp_data / "tracks"
    temp_annotations = temp_data / "annotations"
    temp_videos.mkdir(parents=True)
    temp_tracks.mkdir(parents=True)
    temp_annotations.mkdir(parents=True)

    monkeypatch.setattr("app.VIDEOS_DIR", temp_videos)
    monkeypatch.setattr("app.TRACKS_DIR", temp_tracks)
    monkeypatch.setattr("app.ANNOTATIONS_DIR", temp_annotations)

    yield


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def sample_detections():
    # Two tracks across frames 0..3
    dets = [
        Detection(frame_idx=0, track_id=1, x1=100, y1=100, x2=150, y2=200, confidence=0.9),
        Detection(frame_idx=2, track_id=1, x1=105, y1=98, x2=155, y2=198, confidence=0.88),
        Detection(frame_idx=0, track_id=2, x1=300, y1=120, x2=340, y2=220, confidence=0.85),
        Detection(frame_idx=3, track_id=2, x1=305, y1=125, x2=345, y2=225, confidence=0.86),
    ]
    return dets


@pytest.fixture
def sample_annotations():
    # Simple square pitch points that map to normalized corners
    annotations = [
        {
            "frame_idx": 0,
            "points": [
                {"pitch_id": "corner_tl", "x_img": 0, "y_img": 0},
                {"pitch_id": "corner_tr", "x_img": 400, "y_img": 0},
                {"pitch_id": "corner_bl", "x_img": 0, "y_img": 800},
                {"pitch_id": "corner_br", "x_img": 400, "y_img": 800},
            ]
        }
    ]
    # Build PitchAnnotation objects for programmatic uses
    ann_objs = [PitchAnnotation(**a) for a in annotations]
    return ann_objs


@pytest.fixture
def bad_annotations():
    # Less than 4 points
    annotations = [
        {
            "frame_idx": 0,
            "points": [
                {"pitch_id": "corner_tl", "x_img": 0, "y_img": 0},
                {"pitch_id": "corner_tr", "x_img": 400, "y_img": 0},
            ]
        }
    ]
    return [PitchAnnotation(**a) for a in annotations]


@pytest.fixture
def sample_homography():
    # Identity-like homography mapping image coords directly to pitch-normalized space
    H = np.eye(3, dtype=np.float32)
    return {0: H}


@pytest.fixture
def sample_positions(sample_detections, sample_homography):
    # Map bottom-center manually using identity H: bottom center = (x_center, y_bottom)
    positions = []
    for d in sample_detections:
        x_center = (d.x1 + d.x2) / 2
        y_bottom = d.y2
        positions.append(PlayerPitchPosition(
            frame_idx=d.frame_idx,
            track_id=d.track_id,
            x_pitch=float(x_center),
            y_pitch=float(y_bottom),
            source="homography"
        ))
    return positions


@pytest.fixture
def sample_video_metadata(monkeypatch):
    def fake_metadata(path):
        return {
            "fps": 25,
            "num_frames": 10,
            "width": 1920,
            "height": 1080,
            "duration_seconds": 0.4
        }
    # Patch both pipeline.video.get_video_metadata and the name imported into app
    monkeypatch.setattr("pipeline.video.get_video_metadata", fake_metadata)
    monkeypatch.setattr("app.get_video_metadata", fake_metadata)
    return fake_metadata


@pytest.fixture
def mock_gpu_tracking(monkeypatch, sample_detections):
    """Mock GPU tracking to return sample detections."""
    # Patch run_tracking in detect module to return sample detections
    monkeypatch.setattr("pipeline.detect.run_tracking", lambda path: sample_detections)
    return sample_detections


@pytest.fixture(autouse=True)
def reset_gpu_client():
    """Reset GPU client singleton between tests."""
    import gpu_inference
    gpu_inference._gpu_client = None
    yield
    gpu_inference._gpu_client = None


@pytest.fixture
def mock_modal_response():
    """Sample Modal API response."""
    return {
        "status": "success",
        "detections": [
            {
                "frame_idx": 0,
                "track_id": 1,
                "x1": 100.0,
                "y1": 100.0,
                "x2": 150.0,
                "y2": 200.0,
                "confidence": 0.9
            },
            {
                "frame_idx": 2,
                "track_id": 1,
                "x1": 105.0,
                "y1": 98.0,
                "x2": 155.0,
                "y2": 198.0,
                "confidence": 0.88
            }
        ],
        "count": 2
    }

