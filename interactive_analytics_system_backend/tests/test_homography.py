"""Tests for homography computation and pixel-to-pitch mapping.

The canonical coordinate system is PITCH CANVAS PIXELS (e.g., 850 × 1400).
All tests verify coordinates in this space, not meters.
"""
import numpy as np
import pytest
from pipeline.homography import (
    compute_homography,
    compute_homographies_from_annotations,
    map_pixel_to_pitch,
    resolve_pitch_coordinates,
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
    x, y = map_pixel_to_pitch(425.0, 700.0, H, out_w=850, out_h=1400, k1=0)
    assert abs(x - 425.0) < 1e-3
    assert abs(y - 700.0) < 1e-3


def test_map_pixel_to_pitch_with_distortion():
    """Test that radial distortion is applied."""
    H = np.eye(3, dtype=np.float32)
    # At center of canvas, distortion should be zero
    cx, cy = 425.0, 700.0
    x, y = map_pixel_to_pitch(cx, cy, H, out_w=850, out_h=1400, k1=1e-6)
    assert abs(x - cx) < 1e-3
    assert abs(y - cy) < 1e-3

    # Away from center, distortion should have an effect
    x, y = map_pixel_to_pitch(100.0, 100.0, H, out_w=850, out_h=1400, k1=1e-6)
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

    # Image pixel (4250, 7000) should map to pitch canvas (425, 700)
    x, y = map_pixel_to_pitch(4250.0, 7000.0, H, out_w=850, out_h=1400, k1=0)
    assert abs(x - 425.0) < 1e-3
    assert abs(y - 700.0) < 1e-3


class TestResolvePitchCoordinates:
    """Tests for resolve_pitch_coordinates()."""

    def test_known_vertex_name(self):
        """Named vertices resolve to their stored meter coordinates."""
        x, y = resolve_pitch_coordinates("corner_tl")
        assert x == 0.0
        assert y == 0.0

    def test_known_vertex_name_mid(self):
        """A non-corner named vertex resolves correctly."""
        x, y = resolve_pitch_coordinates("left_45m_line_top")
        assert x == 0.0
        assert y == 45.0

    def test_line_pitch_id_format(self):
        """line_{name}_x{X}_y{Y} format is parsed correctly."""
        x, y = resolve_pitch_coordinates("line_45m_top_x25.3_y45.0")
        assert abs(x - 25.3) < 1e-9
        assert abs(y - 45.0) < 1e-9

    def test_line_pitch_id_integer_coords(self):
        """Integer coordinates in the line format are parsed correctly."""
        x, y = resolve_pitch_coordinates("line_halfway_x42_y70")
        assert abs(x - 42.0) < 1e-9
        assert abs(y - 70.0) < 1e-9

    def test_line_pitch_id_negative_coords(self):
        """Negative coordinates in the line format are parsed correctly."""
        x, y = resolve_pitch_coordinates("line_sideline_x-1.5_y20.0")
        assert abs(x - (-1.5)) < 1e-9
        assert abs(y - 20.0) < 1e-9

    def test_unknown_format_raises(self):
        """Unrecognised pitch_id raises ValueError."""
        with pytest.raises(ValueError):
            resolve_pitch_coordinates("not_a_real_vertex")

    def test_compute_homographies_with_line_pitch_ids(self, sample_annotations):
        """compute_homographies_from_annotations accepts line_{name}_x{X}_y{Y} ids."""
        from pipeline.schemas import PitchAnnotation, PitchPoint
        # Augment the sample annotation with a line-format point
        ann = PitchAnnotation(
            frame_idx=0,
            points=[
                PitchPoint(pitch_id="corner_tl", x_img=0, y_img=0),
                PitchPoint(pitch_id="corner_tr", x_img=400, y_img=0),
                PitchPoint(pitch_id="corner_bl", x_img=0, y_img=800),
                # Line point instead of a named vertex
                PitchPoint(pitch_id="line_45m_top_x42.5_y45.0", x_img=200, y_img=400),
            ]
        )
        homogs = compute_homographies_from_annotations({0: ann.points})
        assert isinstance(homogs, dict)
        assert 0 in homogs
        assert homogs[0].shape == (3, 3)
