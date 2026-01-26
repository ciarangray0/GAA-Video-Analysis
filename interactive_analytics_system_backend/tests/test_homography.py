"""Tests for homography computation and pixel-to-pitch mapping.

The canonical coordinate system is PITCH CANVAS PIXELS (e.g., 850 × 1450).
All tests verify coordinates in this space, not meters.
"""
import numpy as np
from pipeline.homography import (
    compute_homography,
    compute_homographies_from_annotations,
    map_pixel_to_pitch,
)
from pipeline.config import OUT_W, OUT_H, K1


def test_compute_homographies_with_valid_annotations(sample_annotations):
    """Test that homographies are computed from valid annotations."""
    annotations_dict = {ann.frame_idx: ann.points for ann in sample_annotations}
    homogs = compute_homographies_from_annotations(annotations_dict)
    assert isinstance(homogs, dict)
    assert 0 in homogs
    H = homogs[0]
    assert H.shape == (3, 3)


def test_map_pixel_to_pitch_identity():
    """Test mapping with identity homography."""
    H = np.eye(3, dtype=np.float32)
    # With identity H, input coords should equal output (before distortion)
    # With k1=0, no distortion is applied
    x, y = map_pixel_to_pitch(425.0, 725.0, H, out_w=850, out_h=1450, k1=0)
    assert abs(x - 425.0) < 1e-3
    assert abs(y - 725.0) < 1e-3


def test_map_pixel_to_pitch_with_distortion():
    """Test that radial distortion is applied."""
    H = np.eye(3, dtype=np.float32)
    # At center of canvas, distortion should be zero
    cx, cy = 425.0, 725.0
    x, y = map_pixel_to_pitch(cx, cy, H, out_w=850, out_h=1450, k1=1e-6)
    assert abs(x - cx) < 1e-3
    assert abs(y - cy) < 1e-3

    # Away from center, distortion should have an effect
    x, y = map_pixel_to_pitch(100.0, 100.0, H, out_w=850, out_h=1450, k1=1e-6)
    # With k1 > 0, points should move away from center
    # This is barrel distortion effect
    assert x != 100.0 or y != 100.0


def test_homography_maps_to_canvas_pixels():
    """Test that homography output is in pitch canvas pixel space."""
    # Create a simple scaling homography (0.1x scale)
    # This simulates image coords 0-8500 → pitch canvas 0-850
    H = np.array([
        [0.1, 0, 0],
        [0, 0.1, 0],
        [0, 0, 1]
    ], dtype=np.float32)

    # Image pixel (4250, 7250) should map to pitch canvas (425, 725)
    x, y = map_pixel_to_pitch(4250.0, 7250.0, H, out_w=850, out_h=1450, k1=0)
    assert abs(x - 425.0) < 1e-3
    assert abs(y - 725.0) < 1e-3
