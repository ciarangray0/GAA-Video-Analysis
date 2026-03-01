"""Tests for pipeline/frame_mappings.py and related API endpoints."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List
import sys

import numpy as np
import pytest

# Ensure project root is on sys.path so `pipeline` imports work when tests run
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Stub out ultralytics so we don't need the heavy ML dependency
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

from fastapi.testclient import TestClient
from app import app
from store import store
import pipeline.frame_mappings as fm


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_store():
    """Reset in-memory store between tests."""
    store.videos.clear()
    store.detections_cache.clear()
    store.homographies_cache.clear()
    store.homographies_cache_per_frame.clear()
    store.jobs.clear()
    yield
    store.videos.clear()
    store.detections_cache.clear()
    store.homographies_cache.clear()
    store.homographies_cache_per_frame.clear()
    store.jobs.clear()


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def identity_H():
    return np.eye(3, dtype=np.float64)


@pytest.fixture
def simple_H():
    """A simple affine-like homography (2× scale, small translation)."""
    H = np.array(
        [[2.0, 0.0, 10.0], [0.0, 2.0, 20.0], [0.0, 0.0, 1.0]], dtype=np.float64
    )
    return H


@pytest.fixture
def anchor_frames_dict(identity_H):
    """Minimal anchor_frames structure with a known H."""
    return {
        0: {
            "keypoints_image": [[0.0, 0.0], [100.0, 0.0], [100.0, 100.0], [0.0, 100.0]],
            "keypoints_canvas": [[0.0, 0.0], [85.0, 0.0], [85.0, 140.0], [0.0, 140.0]],
            "H": identity_H,
        }
    }


@pytest.fixture
def video_in_store(tmp_path):
    """Register a fake video entry in the store and return its video_id."""
    import uuid

    video_id = str(uuid.uuid4())
    fake_path = tmp_path / f"{video_id}.mp4"
    fake_path.write_bytes(b"fake")
    store.videos[video_id] = {
        "path": str(fake_path),
        "fps": 25,
        "num_frames": 10,
    }
    return video_id


# ---------------------------------------------------------------------------
# Unit tests: solve_homography_from_propagated_points
# ---------------------------------------------------------------------------

class TestSolveHomographyFromPropagatedPoints:
    def test_returns_3x3_matrix(self):
        src = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=np.float32)
        dst = np.array([[0, 0], [85, 0], [85, 140], [0, 140]], dtype=np.float32)
        H = fm.solve_homography_from_propagated_points(src, dst)
        assert H is not None
        assert H.shape == (3, 3)

    def test_returns_none_for_fewer_than_4_points(self):
        src = np.array([[0, 0], [100, 0], [100, 100]], dtype=np.float32)
        dst = np.array([[0, 0], [85, 0], [85, 140]], dtype=np.float32)
        H = fm.solve_homography_from_propagated_points(src, dst)
        assert H is None

    def test_confidence_filtering(self):
        """Low-confidence points are removed before solving."""
        src = np.array(
            [[0, 0], [100, 0], [100, 100], [0, 100], [50, 50]], dtype=np.float32
        )
        dst = np.array(
            [[0, 0], [85, 0], [85, 140], [0, 140], [9999, 9999]], dtype=np.float32
        )
        # Fifth point has zero confidence and should be excluded
        conf = np.array([1.0, 1.0, 1.0, 1.0, 0.0], dtype=np.float32)
        H = fm.solve_homography_from_propagated_points(src, dst, confidence=conf)
        assert H is not None
        assert H.shape == (3, 3)

    def test_all_low_confidence_returns_none(self):
        src = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=np.float32)
        dst = np.array([[0, 0], [85, 0], [85, 140], [0, 140]], dtype=np.float32)
        conf = np.zeros(4, dtype=np.float32)
        H = fm.solve_homography_from_propagated_points(src, dst, confidence=conf)
        assert H is None


# ---------------------------------------------------------------------------
# Unit tests: detect_zoom_from_motion_vectors
# ---------------------------------------------------------------------------

class TestDetectZoomFromMotionVectors:
    def test_single_frame_returns_one(self):
        frame = np.zeros((100, 100), dtype=np.uint8)
        scale = fm.detect_zoom_from_motion_vectors([frame])
        assert scale == 1.0

    def test_identical_frames_returns_near_one(self):
        frame = np.random.randint(0, 255, (200, 200), dtype=np.uint8)
        scale = fm.detect_zoom_from_motion_vectors([frame, frame])
        assert 0.8 <= scale <= 1.2

    def test_returns_float(self):
        a = np.zeros((100, 100), dtype=np.uint8)
        b = np.zeros((100, 100), dtype=np.uint8)
        result = fm.detect_zoom_from_motion_vectors([a, b])
        assert isinstance(result, float)

    def test_clamped_to_sane_range(self):
        a = np.random.randint(0, 255, (200, 200), dtype=np.uint8)
        b = np.random.randint(0, 255, (200, 200), dtype=np.uint8)
        scale = fm.detect_zoom_from_motion_vectors([a, b])
        assert 0.1 <= scale <= 10.0


# ---------------------------------------------------------------------------
# Unit tests: parametric_ptz_estimation
# ---------------------------------------------------------------------------

class TestParametricPtzEstimation:
    def test_single_frame_returns_zero_motion(self):
        frame = np.zeros((200, 200), dtype=np.uint8)
        result = fm.parametric_ptz_estimation([frame])
        assert len(result) == 1
        assert result[0] == {"pan": 0.0, "tilt": 0.0, "zoom": 1.0}

    def test_returns_same_length_as_input(self):
        frames = [np.random.randint(0, 255, (200, 200), dtype=np.uint8) for _ in range(5)]
        result = fm.parametric_ptz_estimation(frames)
        assert len(result) == 5

    def test_first_frame_is_zero_motion(self):
        frames = [np.random.randint(0, 255, (200, 200), dtype=np.uint8) for _ in range(3)]
        result = fm.parametric_ptz_estimation(frames)
        assert result[0]["pan"] == 0.0
        assert result[0]["tilt"] == 0.0
        assert result[0]["zoom"] == 1.0

    def test_all_entries_have_required_keys(self):
        frames = [np.random.randint(0, 255, (200, 200), dtype=np.uint8) for _ in range(3)]
        result = fm.parametric_ptz_estimation(frames)
        for entry in result:
            assert "pan" in entry
            assert "tilt" in entry
            assert "zoom" in entry


# ---------------------------------------------------------------------------
# Unit tests: compare_homography_heatmap
# ---------------------------------------------------------------------------

class TestCompareHomographyHeatmap:
    def test_identity_vs_identity_has_zero_displacement(self):
        H = np.eye(3, dtype=np.float64)
        result = fm.compare_homography_heatmap(H, H)
        assert result["mean_displacement"] == pytest.approx(0.0, abs=1e-6)
        assert result["max_displacement"] == pytest.approx(0.0, abs=1e-6)

    def test_returns_expected_keys(self):
        H = np.eye(3, dtype=np.float64)
        result = fm.compare_homography_heatmap(H, H)
        assert "mean_displacement" in result
        assert "max_displacement" in result
        assert "grid_displacements" in result
        assert "heatmap" in result

    def test_heatmap_is_base64_string(self):
        import base64

        H1 = np.eye(3, dtype=np.float64)
        H2 = np.array([[2, 0, 10], [0, 2, 20], [0, 0, 1]], dtype=np.float64)
        result = fm.compare_homography_heatmap(H1, H2)
        assert isinstance(result["heatmap"], str)
        # Should be valid base64
        base64.b64decode(result["heatmap"])

    def test_grid_displacements_length(self):
        H = np.eye(3, dtype=np.float64)
        grid_size = 10
        result = fm.compare_homography_heatmap(H, H, grid_size=grid_size)
        assert len(result["grid_displacements"]) == grid_size ** 2

    def test_nonzero_displacement_for_different_Hs(self):
        H1 = np.eye(3, dtype=np.float64)
        H2 = np.array([[1, 0, 50], [0, 1, 100], [0, 0, 1]], dtype=np.float64)
        result = fm.compare_homography_heatmap(H1, H2)
        assert result["mean_displacement"] > 0.0


# ---------------------------------------------------------------------------
# Unit tests: _generate_interpolated (internal helper)
# ---------------------------------------------------------------------------

class TestGenerateInterpolated:
    def test_exact_anchor_frames_preserved(self, identity_H, simple_H):
        anchor_frames = {
            0: {"H": identity_H},
            9: {"H": simple_H},
        }
        result = fm._generate_interpolated([0, 9], anchor_frames, num_frames=10)
        np.testing.assert_array_almost_equal(result[0], identity_H)
        np.testing.assert_array_almost_equal(result[9], simple_H)

    def test_intermediate_frame_is_interpolated(self, identity_H, simple_H):
        anchor_frames = {
            0: {"H": identity_H},
            10: {"H": simple_H},
        }
        result = fm._generate_interpolated([0, 10], anchor_frames, num_frames=11)
        midpoint = result[5]
        expected = 0.5 * identity_H + 0.5 * simple_H
        np.testing.assert_array_almost_equal(midpoint, expected)

    def test_no_H_returns_empty(self):
        anchor_frames = {0: {"keypoints_image": []}}
        result = fm._generate_interpolated([0], anchor_frames, num_frames=5)
        assert result == {}

    def test_before_first_anchor_clamps_to_first(self, identity_H):
        anchor_frames = {5: {"H": identity_H}}
        result = fm._generate_interpolated([5], anchor_frames, num_frames=10)
        np.testing.assert_array_almost_equal(result[0], identity_H)

    def test_after_last_anchor_clamps_to_last(self, simple_H):
        anchor_frames = {0: {"H": simple_H}}
        result = fm._generate_interpolated([0], anchor_frames, num_frames=5)
        np.testing.assert_array_almost_equal(result[4], simple_H)


# ---------------------------------------------------------------------------
# Unit tests: generate_per_frame_mappings (job creation)
# ---------------------------------------------------------------------------

class TestGeneratePerFrameMappings:
    def test_returns_job_id_string(self, video_in_store, anchor_frames_dict):
        job_id = fm.generate_per_frame_mappings(
            video_in_store, anchor_frames_dict, method="interpolate"
        )
        assert isinstance(job_id, str)
        assert len(job_id) > 0

    def test_job_created_in_store(self, video_in_store, anchor_frames_dict):
        job_id = fm.generate_per_frame_mappings(
            video_in_store, anchor_frames_dict, method="interpolate"
        )
        assert job_id in store.jobs

    def test_job_has_required_fields(self, video_in_store, anchor_frames_dict):
        job_id = fm.generate_per_frame_mappings(
            video_in_store, anchor_frames_dict, method="interpolate"
        )
        job = store.jobs[job_id]
        assert "job_id" in job
        assert "video_id" in job
        assert "status" in job
        assert "method" in job
        assert "progress" in job
        assert "total" in job

    def test_job_completed_for_interpolate(self, video_in_store, anchor_frames_dict):
        """Interpolate method should complete quickly (no I/O needed)."""
        import time

        job_id = fm.generate_per_frame_mappings(
            video_in_store, anchor_frames_dict, method="interpolate"
        )
        # Give the background thread up to 2 seconds to finish
        deadline = time.time() + 2.0
        while store.jobs[job_id]["status"] not in ("completed", "failed"):
            if time.time() > deadline:
                break
            time.sleep(0.05)

        assert store.jobs[job_id]["status"] == "completed"

    def test_per_frame_cache_populated_after_interpolate(self, video_in_store, anchor_frames_dict):
        import time

        job_id = fm.generate_per_frame_mappings(
            video_in_store, anchor_frames_dict, method="interpolate"
        )
        deadline = time.time() + 2.0
        while store.jobs[job_id]["status"] not in ("completed", "failed"):
            if time.time() > deadline:
                break
            time.sleep(0.05)

        assert store.jobs[job_id]["status"] == "completed"
        assert video_in_store in store.homographies_cache_per_frame
        cache = store.homographies_cache_per_frame[video_in_store]
        assert len(cache) > 0

    def test_missing_video_fails_job(self, anchor_frames_dict):
        import time

        job_id = fm.generate_per_frame_mappings(
            "nonexistent_video_id", anchor_frames_dict, method="interpolate"
        )
        deadline = time.time() + 2.0
        while store.jobs[job_id]["status"] not in ("completed", "failed"):
            if time.time() > deadline:
                break
            time.sleep(0.05)

        assert store.jobs[job_id]["status"] == "failed"


# ---------------------------------------------------------------------------
# API endpoint tests
# ---------------------------------------------------------------------------

class TestJobEndpoints:
    def test_list_jobs_empty(self, client):
        resp = client.get("/api/jobs/")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_list_jobs_returns_all_jobs(self, client, video_in_store, anchor_frames_dict):
        fm.generate_per_frame_mappings(
            video_in_store, anchor_frames_dict, method="interpolate"
        )
        resp = client.get("/api/jobs/")
        assert resp.status_code == 200
        jobs = resp.json()
        assert len(jobs) == 1

    def test_get_job_not_found(self, client):
        resp = client.get("/api/jobs/does-not-exist")
        assert resp.status_code == 404

    def test_get_job_found(self, client, video_in_store, anchor_frames_dict):
        job_id = fm.generate_per_frame_mappings(
            video_in_store, anchor_frames_dict, method="interpolate"
        )
        resp = client.get(f"/api/jobs/{job_id}")
        assert resp.status_code == 200
        body = resp.json()
        assert body["job_id"] == job_id
        assert body["video_id"] == video_in_store


class TestPerFrameMappingEndpoint:
    """Tests for POST/GET /videos/{video_id}/per-frame-mappings."""

    def test_start_per_frame_mappings_returns_job(
        self, client, sample_video_metadata, video_in_store
    ):
        payload = {
            "anchor_frames": {
                "0": {
                    "keypoints_image": [[0, 0], [100, 0], [100, 100], [0, 100]],
                    "keypoints_canvas": [[0, 0], [85, 0], [85, 140], [0, 140]],
                    "H": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                }
            },
            "method": "interpolate",
            "options": {},
        }
        resp = client.post(f"/videos/{video_in_store}/per-frame-mappings", json=payload)
        assert resp.status_code == 200
        body = resp.json()
        assert "job_id" in body
        assert body["video_id"] == video_in_store
        assert body["status"] in ("queued", "running", "completed")

    def test_start_per_frame_mappings_404_for_unknown_video(self, client):
        payload = {
            "anchor_frames": {},
            "method": "interpolate",
            "options": {},
        }
        resp = client.post("/videos/nonexistent/per-frame-mappings", json=payload)
        assert resp.status_code == 404

    def test_get_per_frame_mapping_after_job(
        self, client, sample_video_metadata, video_in_store
    ):
        """After interpolate job completes, per-frame mapping should be retrievable."""
        import time

        payload = {
            "anchor_frames": {
                "0": {
                    "keypoints_image": [[0, 0], [100, 0], [100, 100], [0, 100]],
                    "keypoints_canvas": [[0, 0], [85, 0], [85, 140], [0, 140]],
                    "H": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                }
            },
            "method": "interpolate",
            "options": {},
        }
        r = client.post(f"/videos/{video_in_store}/per-frame-mappings", json=payload)
        assert r.status_code == 200
        job_id = r.json()["job_id"]

        # Poll until the job finishes
        deadline = time.time() + 3.0
        while True:
            jr = client.get(f"/api/jobs/{job_id}")
            if jr.json()["status"] in ("completed", "failed") or time.time() > deadline:
                break
            time.sleep(0.05)

        assert jr.json()["status"] == "completed"

        resp = client.get(f"/videos/{video_in_store}/per-frame-mappings/0")
        assert resp.status_code == 200
        body = resp.json()
        assert body["frame_idx"] == 0
        assert len(body["H"]) == 3

    def test_get_per_frame_mapping_404_no_cache(
        self, client, sample_video_metadata, video_in_store
    ):
        resp = client.get(f"/videos/{video_in_store}/per-frame-mappings/0")
        assert resp.status_code == 404


class TestCompareHomographiesEndpoint:
    """Tests for POST /videos/{video_id}/compare-homographies."""

    def test_compare_identity_matrices(self, client, sample_video_metadata, video_in_store):
        payload = {
            "H1": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            "H2": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            "grid_size": 5,
        }
        resp = client.post(f"/videos/{video_in_store}/compare-homographies", json=payload)
        assert resp.status_code == 200
        body = resp.json()
        assert body["mean_displacement"] == pytest.approx(0.0, abs=1e-5)

    def test_compare_different_matrices(self, client, sample_video_metadata, video_in_store):
        payload = {
            "H1": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            "H2": [[1, 0, 50], [0, 1, 100], [0, 0, 1]],
            "grid_size": 5,
        }
        resp = client.post(f"/videos/{video_in_store}/compare-homographies", json=payload)
        assert resp.status_code == 200
        body = resp.json()
        assert body["mean_displacement"] > 0.0

    def test_invalid_H_shape_returns_400(self, client, sample_video_metadata, video_in_store):
        payload = {
            "H1": [[1, 0], [0, 1]],  # 2×2, invalid
            "H2": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            "grid_size": 5,
        }
        resp = client.post(f"/videos/{video_in_store}/compare-homographies", json=payload)
        assert resp.status_code == 400

    def test_404_for_unknown_video(self, client):
        payload = {
            "H1": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            "H2": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        }
        resp = client.post("/videos/nonexistent/compare-homographies", json=payload)
        assert resp.status_code == 404
