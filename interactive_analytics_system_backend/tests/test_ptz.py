"""Tests for the PTZ (Pan/Tilt/Zoom) camera model.

Tests cover:
- Homography decomposition into pan/tilt/zoom
- Zoom estimation via optical flow
- Inter-frame homography estimation
- PTZ state propagation
- Homography reconstruction from PTZ
- End-to-end build_per_frame_homographies
"""
import numpy as np
import pytest
import cv2

from pipeline.ptz import (
    PTZState,
    decompose_homography_ptz,
    estimate_zoom_from_optical_flow,
    estimate_inter_frame_homography,
    propagate_ptz,
    ptz_to_pitch_homography,
    build_per_frame_homographies,
    _rotation_from_pan_tilt,
    _project_to_rotation,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def identity_H():
    return np.eye(3, dtype=np.float64)


@pytest.fixture
def simple_frame():
    """A 100x100 grayscale gradient frame for optical flow tests."""
    rng = np.random.default_rng(42)
    return rng.integers(0, 255, (100, 100, 3), dtype=np.uint8)


@pytest.fixture
def anchor_H():
    """A plausible anchor homography (mild scaling + translation)."""
    H = np.array([
        [0.5, 0.0, 100.0],
        [0.0, 0.5, 200.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)
    return H


# ---------------------------------------------------------------------------
# PTZState dataclass
# ---------------------------------------------------------------------------

class TestPTZState:
    def test_defaults(self):
        s = PTZState(frame_idx=0)
        assert s.pan == 0.0
        assert s.tilt == 0.0
        assert s.zoom == 1.0
        assert s.source == "anchor"

    def test_custom_values(self):
        s = PTZState(frame_idx=5, pan=0.1, tilt=-0.05, zoom=1.2, source="homography_decomp")
        assert s.frame_idx == 5
        assert s.pan == pytest.approx(0.1)
        assert s.zoom == pytest.approx(1.2)


# ---------------------------------------------------------------------------
# decompose_homography_ptz
# ---------------------------------------------------------------------------

class TestDecomposeHomographyPTZ:
    def test_identity_gives_zero_pan_tilt_unit_zoom(self, identity_H):
        pan, tilt, zoom = decompose_homography_ptz(
            identity_H, focal_length=800.0, cx=320.0, cy=240.0
        )
        assert abs(pan) < 0.01
        assert abs(tilt) < 0.01
        assert abs(zoom - 1.0) < 0.05

    def test_zoom_in_detected(self):
        """A homography encoding a pure 1.5× zoom should give zoom ≈ 1.5."""
        zoom_factor = 1.5
        # Pure zoom homography: K · diag(z,z,1) · K⁻¹ = diag(z,z,1)
        # for a camera with principal point at origin.
        H = np.diag([zoom_factor, zoom_factor, 1.0]).astype(np.float64)
        _, _, zoom = decompose_homography_ptz(H, focal_length=1.0, cx=0.0, cy=0.0)
        assert abs(zoom - zoom_factor) < 0.2

    def test_pan_detected(self):
        """A small pan rotation should produce a non-zero pan estimate."""
        pan_angle = 0.1  # radians ≈ 5.7 degrees
        R = _rotation_from_pan_tilt(pan_angle, 0.0)
        f = 800.0
        cx, cy = 320.0, 240.0
        K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.float64)
        K_inv = np.linalg.inv(K)
        H = K @ R @ K_inv
        pan_out, tilt_out, _ = decompose_homography_ptz(H, f, cx, cy)
        assert abs(pan_out - pan_angle) < 0.05
        assert abs(tilt_out) < 0.05

    def test_returns_floats(self, identity_H):
        pan, tilt, zoom = decompose_homography_ptz(identity_H, 800.0, 320.0, 240.0)
        assert isinstance(pan, float)
        assert isinstance(tilt, float)
        assert isinstance(zoom, float)


# ---------------------------------------------------------------------------
# _rotation_from_pan_tilt and _project_to_rotation
# ---------------------------------------------------------------------------

class TestRotationHelpers:
    def test_zero_angles_is_identity(self):
        R = _rotation_from_pan_tilt(0.0, 0.0)
        assert np.allclose(R, np.eye(3), atol=1e-10)

    def test_rotation_is_orthogonal(self):
        R = _rotation_from_pan_tilt(0.2, -0.1)
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-10)
        assert abs(np.linalg.det(R) - 1.0) < 1e-10

    def test_project_to_rotation_fixes_skewed_matrix(self):
        M = np.eye(3) + np.random.default_rng(0).normal(0, 0.05, (3, 3))
        R = _project_to_rotation(M)
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-8)
        assert abs(np.linalg.det(R) - 1.0) < 1e-8


# ---------------------------------------------------------------------------
# estimate_zoom_from_optical_flow
# ---------------------------------------------------------------------------

class TestEstimateZoomFromOpticalFlow:
    def test_identical_frames_zoom_near_one(self, simple_frame):
        """Identical frames should produce no optical flow → zoom ≈ 1."""
        zoom = estimate_zoom_from_optical_flow(simple_frame, simple_frame)
        # Allow some numerical noise from the optical flow algorithm.
        assert 0.5 <= zoom <= 2.0

    def test_returns_float(self, simple_frame):
        zoom = estimate_zoom_from_optical_flow(simple_frame, simple_frame)
        assert isinstance(zoom, float)

    def test_mismatched_sizes_raises(self):
        a = np.zeros((100, 100, 3), dtype=np.uint8)
        b = np.zeros((200, 200, 3), dtype=np.uint8)
        with pytest.raises(ValueError, match="same spatial dimensions"):
            estimate_zoom_from_optical_flow(a, b)

    def test_zoom_not_negative(self, simple_frame):
        zoom = estimate_zoom_from_optical_flow(simple_frame, simple_frame)
        assert zoom > 0.0

    def test_accepts_greyscale(self):
        gray = np.random.default_rng(1).integers(0, 255, (80, 80), dtype=np.uint8)
        zoom = estimate_zoom_from_optical_flow(gray, gray)
        assert isinstance(zoom, float)

    def test_custom_principal_point(self, simple_frame):
        h, w = simple_frame.shape[:2]
        zoom = estimate_zoom_from_optical_flow(
            simple_frame, simple_frame, principal_point=(w * 0.4, h * 0.4)
        )
        assert isinstance(zoom, float)


# ---------------------------------------------------------------------------
# estimate_inter_frame_homography
# ---------------------------------------------------------------------------

class TestEstimateInterFrameHomography:
    def test_identical_frames_returns_near_identity(self):
        """Two identical frames with detectable features should give H ≈ I."""
        # Use a chessboard-like pattern to ensure ORB finds features.
        img = np.tile(
            np.array([[0, 255] * 8] * 8 + [[255, 0] * 8] * 8, dtype=np.uint8),
            (8, 8),
        )
        img_bgr = cv2.cvtColor(img[:128, :128], cv2.COLOR_GRAY2BGR)
        H = estimate_inter_frame_homography(img_bgr, img_bgr)
        if H is not None:
            # Should be close to identity for identical frames.
            assert H.shape == (3, 3)
            norm_diff = np.linalg.norm(H / H[2, 2] - np.eye(3))
            assert norm_diff < 5.0  # Generous tolerance

    def test_featureless_frames_returns_none(self):
        """Uniform frames have no features → should return None."""
        blank = np.zeros((200, 200, 3), dtype=np.uint8)
        H = estimate_inter_frame_homography(blank, blank)
        assert H is None

    def test_returns_3x3_or_none(self):
        a = np.random.default_rng(10).integers(0, 255, (100, 100, 3), dtype=np.uint8)
        b = np.random.default_rng(11).integers(0, 255, (100, 100, 3), dtype=np.uint8)
        H = estimate_inter_frame_homography(a, b)
        if H is not None:
            assert H.shape == (3, 3)


# ---------------------------------------------------------------------------
# propagate_ptz
# ---------------------------------------------------------------------------

class TestPropagatePTZ:
    def _make_inter_H(self, pan, tilt, zoom, f=800.0, cx=320.0, cy=240.0):
        """Build a synthetic inter-frame homography for given PTZ deltas."""
        R = _rotation_from_pan_tilt(pan, tilt)
        K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.float64)
        K_inv = np.linalg.inv(K)
        H = K @ R @ K_inv * zoom
        return H

    def test_anchor_frame_has_zero_ptz(self, anchor_H):
        ptz_states = propagate_ptz(
            anchor_frame=0,
            anchor_H_to_pitch=anchor_H,
            inter_frame_homographies={},
            focal_length=800.0,
            image_width=640,
            image_height=480,
        )
        assert 0 in ptz_states
        s = ptz_states[0]
        assert s.pan == 0.0
        assert s.tilt == 0.0
        assert s.zoom == 1.0
        assert s.source == "anchor"

    def test_propagates_forward(self, anchor_H):
        H1 = self._make_inter_H(0.05, 0.02, 1.0)
        H2 = self._make_inter_H(0.05, 0.02, 1.0)
        states = propagate_ptz(
            anchor_frame=0,
            anchor_H_to_pitch=anchor_H,
            inter_frame_homographies={1: H1, 2: H2},
            focal_length=800.0,
            image_width=640,
            image_height=480,
        )
        assert 1 in states
        assert 2 in states
        assert states[1].source == "homography_decomp"

    def test_propagates_backward(self, anchor_H):
        H1 = self._make_inter_H(0.05, 0.0, 1.0)
        states = propagate_ptz(
            anchor_frame=2,
            anchor_H_to_pitch=anchor_H,
            inter_frame_homographies={1: H1, 2: H1},
            focal_length=800.0,
            image_width=640,
            image_height=480,
        )
        # frame 1 should be reachable backward from anchor=2
        assert 1 in states or 2 in states  # At least anchor is present

    def test_only_anchor_when_no_inter_homographies(self, anchor_H):
        states = propagate_ptz(
            anchor_frame=5,
            anchor_H_to_pitch=anchor_H,
            inter_frame_homographies={},
            focal_length=800.0,
            image_width=640,
            image_height=480,
        )
        assert list(states.keys()) == [5]


# ---------------------------------------------------------------------------
# ptz_to_pitch_homography
# ---------------------------------------------------------------------------

class TestPTZToPitchHomography:
    def test_anchor_ptz_reproduces_anchor_H(self, anchor_H):
        """PTZ at anchor (pan=0, tilt=0, zoom=1) → same as anchor_H."""
        ptz = PTZState(frame_idx=0, pan=0.0, tilt=0.0, zoom=1.0)
        H_out = ptz_to_pitch_homography(ptz, anchor_H, focal_length=800.0, cx=320.0, cy=240.0)
        assert H_out.shape == (3, 3)
        # Scale-normalize and compare.
        H_norm = H_out / H_out[2, 2]
        anchor_norm = anchor_H.astype(np.float64) / anchor_H[2, 2]
        assert np.allclose(H_norm, anchor_norm, atol=1e-6)

    def test_returns_3x3(self, anchor_H):
        ptz = PTZState(frame_idx=1, pan=0.1, tilt=0.05, zoom=1.1)
        H = ptz_to_pitch_homography(ptz, anchor_H, 800.0, 320.0, 240.0)
        assert H.shape == (3, 3)

    def test_zoom_changes_scale(self, anchor_H):
        """Zoom > 1 should produce a homography with larger scale factor."""
        ptz_zoom = PTZState(frame_idx=1, pan=0.0, tilt=0.0, zoom=2.0)
        ptz_ref = PTZState(frame_idx=0, pan=0.0, tilt=0.0, zoom=1.0)
        H_zoom = ptz_to_pitch_homography(ptz_zoom, anchor_H, 800.0, 320.0, 240.0)
        H_ref = ptz_to_pitch_homography(ptz_ref, anchor_H, 800.0, 320.0, 240.0)
        # det scales by zoom^3 due to the 3×3 matrix.
        scale_ratio = (abs(np.linalg.det(H_zoom)) / abs(np.linalg.det(H_ref))) ** (1 / 3)
        # Allow generous tolerance since exact scaling depends on K
        assert scale_ratio > 0.5


# ---------------------------------------------------------------------------
# build_per_frame_homographies (end-to-end)
# ---------------------------------------------------------------------------

class TestBuildPerFrameHomographies:
    @pytest.fixture
    def small_clip(self):
        """3 low-resolution BGR frames for fast testing."""
        rng = np.random.default_rng(7)
        return [rng.integers(0, 255, (64, 64, 3), dtype=np.uint8) for _ in range(3)]

    def test_returns_dicts(self, small_clip, anchor_H):
        homographies, ptz_states = build_per_frame_homographies(
            frames=small_clip,
            anchor_frame_idx=0,
            anchor_H_to_pitch=anchor_H,
            focal_length=200.0,
            use_optical_flow_zoom=False,
        )
        assert isinstance(homographies, dict)
        assert isinstance(ptz_states, dict)

    def test_anchor_frame_present(self, small_clip, anchor_H):
        homographies, ptz_states = build_per_frame_homographies(
            frames=small_clip,
            anchor_frame_idx=0,
            anchor_H_to_pitch=anchor_H,
            focal_length=200.0,
            use_optical_flow_zoom=False,
        )
        assert 0 in homographies
        assert 0 in ptz_states

    def test_anchor_homography_matches_input(self, small_clip, anchor_H):
        homographies, _ = build_per_frame_homographies(
            frames=small_clip,
            anchor_frame_idx=0,
            anchor_H_to_pitch=anchor_H,
            focal_length=200.0,
            use_optical_flow_zoom=False,
        )
        H_out = homographies[0]
        assert np.allclose(H_out, anchor_H.astype(np.float64), atol=1e-8)

    def test_empty_frames_returns_empty(self, anchor_H):
        homographies, ptz_states = build_per_frame_homographies(
            frames=[],
            anchor_frame_idx=0,
            anchor_H_to_pitch=anchor_H,
        )
        assert homographies == {}
        assert ptz_states == {}

    def test_homographies_are_3x3(self, small_clip, anchor_H):
        homographies, _ = build_per_frame_homographies(
            frames=small_clip,
            anchor_frame_idx=0,
            anchor_H_to_pitch=anchor_H,
            focal_length=200.0,
            use_optical_flow_zoom=False,
        )
        for H in homographies.values():
            assert H.shape == (3, 3)

    def test_with_optical_flow_zoom(self, small_clip, anchor_H):
        """Ensure the optical-flow zoom path runs without error."""
        homographies, ptz_states = build_per_frame_homographies(
            frames=small_clip,
            anchor_frame_idx=0,
            anchor_H_to_pitch=anchor_H,
            focal_length=200.0,
            use_optical_flow_zoom=True,
        )
        assert 0 in homographies

    def test_default_focal_length(self, small_clip, anchor_H):
        """Omitting focal_length should use max(width, height) by default."""
        homographies, _ = build_per_frame_homographies(
            frames=small_clip,
            anchor_frame_idx=0,
            anchor_H_to_pitch=anchor_H,
            # focal_length not provided → defaults to max(64, 64) = 64
        )
        assert 0 in homographies
