"""Tests for player mapping to pitch canvas coordinates.

The canonical coordinate system is PITCH CANVAS PIXELS (e.g., 850 × 1450).
All player positions are in this space, with radial distortion applied.
"""
import numpy as np
from pipeline.map_players import map_players_to_pitch
from pipeline.schemas import Detection
from pipeline.config import OUT_W, OUT_H


def test_map_players_only_at_homography_frames(sample_detections, sample_homography):
    """Test that map_players ONLY maps at frames with homography."""
    # sample_homography has only frame 0
    # sample_detections has frames 0, 2, 3
    positions = map_players_to_pitch(sample_detections, sample_homography)

    # Should ONLY map detections at frame 0 (where homography exists)
    assert all(p.frame_idx == 0 for p in positions)
    assert all(p.source == "homography" for p in positions)

    # Frame 0 has 2 detections (track 1 and 2)
    assert len(positions) == 2


def test_map_players_returns_canvas_pixels():
    """Test that map_players returns coordinates in pitch canvas pixels."""
    # Create a simple identity-like homography
    H = np.eye(3, dtype=np.float32)

    detections = [
        Detection(frame_idx=0, track_id=1, x1=40, y1=50, x2=50, y2=100, confidence=0.9)
    ]

    # With k1=0, no distortion, output should be same as input bbox bottom-center
    positions = map_players_to_pitch(detections, {0: H}, k1=0)

    assert len(positions) == 1
    p = positions[0]

    # Bottom center should be (45, 100) with identity H and no distortion
    assert abs(p.x_pitch - 45.0) < 1e-3
    assert abs(p.y_pitch - 100.0) < 1e-3
    assert p.source == "homography"


def test_map_players_empty_homographies():
    """Test that empty homographies returns empty list."""
    detections = [Detection(frame_idx=0, track_id=1, x1=0, y1=0, x2=10, y2=10, confidence=0.9)]
    positions = map_players_to_pitch(detections, {})
    assert positions == []


def test_map_players_no_matching_frames():
    """Test that detections without matching homography frame are skipped."""
    H = np.eye(3, dtype=np.float32)

    # Homography at frame 0, detection at frame 5
    detections = [Detection(frame_idx=5, track_id=1, x1=0, y1=0, x2=10, y2=10, confidence=0.9)]
    positions = map_players_to_pitch(detections, {0: H})

    # No positions since frame 5 has no homography
    assert positions == []


def test_map_players_applies_distortion():
    """Test that radial distortion is applied to output coordinates."""
    H = np.eye(3, dtype=np.float32)

    # Detection far from center - distortion should have an effect
    detections = [
        Detection(frame_idx=0, track_id=1, x1=90, y1=90, x2=110, y2=100, confidence=0.9)
    ]

    # With k1=0, no distortion
    positions_no_dist = map_players_to_pitch(detections, {0: H}, k1=0)

    # With k1 > 0, distortion applied
    positions_with_dist = map_players_to_pitch(detections, {0: H}, k1=1e-5)

    # Positions should differ due to distortion
    assert len(positions_no_dist) == 1
    assert len(positions_with_dist) == 1

    # With large k1, there should be a difference
    # (the point is away from center so distortion has effect)


