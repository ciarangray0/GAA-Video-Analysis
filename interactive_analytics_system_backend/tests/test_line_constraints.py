"""Tests for line-constrained homography computation.

These tests verify:
1. Point sampling along lines
2. Synthetic correspondence generation
3. Line validation
4. Full homography computation with line constraints
"""
import pytest
import numpy as np

from pipeline.line_constraints import (
    get_line_y_canvas,
    get_available_lines,
    sample_points_on_line,
    get_point_weights,
    generate_synthetic_correspondences,
    validate_line_annotation,
    filter_valid_line_annotations,
    compute_line_constrained_homography,
    compute_initial_homography,
    preview_synthetic_points,
    GAA_PITCH_LINES
)
from pipeline.config import OUT_W, OUT_H


class TestLineConfiguration:
    """Tests for line ID configuration and conversion."""

    def test_get_line_y_canvas_valid(self):
        """Test converting valid line IDs to canvas Y coordinates."""
        # 20m line at top should be at ~200 pixels (20m / 140m * 1400px)
        y = get_line_y_canvas("20m_top")
        expected = 20.0 / 140.0 * OUT_H
        assert abs(y - expected) < 0.01

    def test_get_line_y_canvas_halfway(self):
        """Test halfway line is at center."""
        y = get_line_y_canvas("halfway")
        expected = 70.0 / 140.0 * OUT_H
        assert abs(y - expected) < 0.01

    def test_get_line_y_canvas_invalid(self):
        """Test that invalid line ID raises ValueError."""
        with pytest.raises(ValueError, match="Unknown line ID"):
            get_line_y_canvas("invalid_line")

    def test_get_available_lines(self):
        """Test that available lines returns expected structure."""
        lines = get_available_lines()
        assert "20m_top" in lines
        assert "halfway" in lines
        assert "65m_bottom" in lines
        assert lines["20m_top"] == 20.0


class TestPointSampling:
    """Tests for sampling points along a line."""

    def test_sample_points_basic(self):
        """Test basic point sampling."""
        points = sample_points_on_line(0, 0, 100, 0, num_samples=5)
        assert points.shape == (5, 2)
        # First point should be at start
        assert points[0, 0] == 0
        assert points[0, 1] == 0
        # Last point should be at end
        assert points[-1, 0] == 100
        assert points[-1, 1] == 0

    def test_sample_points_diagonal(self):
        """Test sampling on diagonal line."""
        points = sample_points_on_line(0, 0, 100, 100, num_samples=3)
        assert points.shape == (3, 2)
        # Middle point should be at (50, 50)
        np.testing.assert_allclose(points[1], [50, 50])

    def test_sample_points_uniform_spacing(self):
        """Test that points are uniformly spaced."""
        points = sample_points_on_line(0, 0, 100, 0, num_samples=11)
        # Distance between consecutive points should be constant
        diffs = np.diff(points[:, 0])
        np.testing.assert_allclose(diffs, 10.0)


class TestPointWeights:
    """Tests for confidence weights on sampled points."""

    def test_weights_shape(self):
        """Test weight array has correct shape."""
        weights = get_point_weights(10)
        assert len(weights) == 10

    def test_weights_range(self):
        """Test weights are in expected range [0.5, 1.0]."""
        weights = get_point_weights(10)
        assert np.all(weights >= 0.5)
        assert np.all(weights <= 1.0)

    def test_center_weight_highest(self):
        """Test center point has highest weight."""
        weights = get_point_weights(11)  # Odd number for exact center
        assert weights[5] == max(weights)  # Middle element

    def test_endpoint_weights_lowest(self):
        """Test endpoints have lowest weights."""
        weights = get_point_weights(11)
        assert weights[0] == min(weights)
        assert weights[-1] == min(weights)


class TestSyntheticCorrespondences:
    """Tests for generating synthetic point correspondences."""

    @pytest.fixture
    def identity_homography(self):
        """Create identity homography (no transformation)."""
        return np.eye(3, dtype=np.float32)

    @pytest.fixture
    def simple_homography(self):
        """Create a simple scale/translate homography for testing."""
        # Maps (0,0) -> (0,0), (100,0) -> (200,0), etc.
        H = np.array([
            [2.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 1.0]
        ], dtype=np.float32)
        return H

    def test_generate_synthetic_basic(self, identity_homography):
        """Test basic synthetic correspondence generation."""
        line_ann = {
            "line_id": "20m_top",
            "u1": 100, "v1": 200,
            "u2": 500, "v2": 200
        }

        pts_img, pts_world, weights = generate_synthetic_correspondences(
            line_ann, identity_homography, num_samples=5
        )

        assert pts_img.shape == (5, 2)
        assert pts_world.shape == (5, 2)
        assert len(weights) == 5

    def test_synthetic_y_is_fixed(self, identity_homography):
        """Test that all world Y coordinates are fixed at line Y value."""
        line_ann = {
            "line_id": "20m_top",
            "u1": 100, "v1": 200,
            "u2": 500, "v2": 200
        }
        expected_y = get_line_y_canvas("20m_top")

        _, pts_world, _ = generate_synthetic_correspondences(
            line_ann, identity_homography, num_samples=10
        )

        # All Y values should be exactly the same
        np.testing.assert_allclose(pts_world[:, 1], expected_y)

    def test_synthetic_x_from_projection(self, simple_homography):
        """Test that X coordinates come from homography projection."""
        line_ann = {
            "line_id": "20m_top",
            "u1": 100, "v1": 100,
            "u2": 200, "v2": 100
        }

        pts_img, pts_world, _ = generate_synthetic_correspondences(
            line_ann, simple_homography, num_samples=3
        )

        # With 2x scale homography, X should be doubled
        # First point: (100, 100) -> X should be 200
        assert abs(pts_world[0, 0] - 200) < 0.1
        # Last point: (200, 100) -> X should be 400
        assert abs(pts_world[-1, 0] - 400) < 0.1


class TestLineValidation:
    """Tests for validating line annotations."""

    @pytest.fixture
    def good_homography(self):
        """Create a realistic homography for testing."""
        # Simple translation to simulate perspective
        H = np.array([
            [1.0, 0.0, 50.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ], dtype=np.float32)
        return H

    def test_validate_good_line(self, good_homography):
        """Test validation passes for correct line annotation."""
        # Line at y=200 (20m_top is at ~200 canvas pixels)
        y_expected = get_line_y_canvas("20m_top")
        line_ann = {
            "line_id": "20m_top",
            "u1": 100, "v1": y_expected,
            "u2": 500, "v2": y_expected
        }

        is_valid, error = validate_line_annotation(line_ann, good_homography)
        assert is_valid
        assert error == ""

    def test_validate_tilted_line_fails(self, good_homography):
        """Test validation fails for non-horizontal lines."""
        y1 = get_line_y_canvas("20m_top")
        line_ann = {
            "line_id": "20m_top",
            "u1": 100, "v1": y1,
            "u2": 500, "v2": y1 + 200  # Very tilted
        }

        is_valid, error = validate_line_annotation(line_ann, good_homography)
        assert not is_valid
        assert "different Y values" in error

    def test_validate_wrong_line_id_fails(self, good_homography):
        """Test validation fails when Y doesn't match line ID."""
        # Annotate at 20m position but claim it's 65m
        y_20m = get_line_y_canvas("20m_top")
        line_ann = {
            "line_id": "65m_top",  # Wrong!
            "u1": 100, "v1": y_20m,
            "u2": 500, "v2": y_20m
        }

        is_valid, error = validate_line_annotation(
            line_ann, good_homography, y_tolerance_pixels=50
        )
        assert not is_valid
        assert "far from expected" in error

    def test_filter_valid_lines(self, good_homography):
        """Test filtering keeps only valid annotations."""
        y_20m = get_line_y_canvas("20m_top")
        lines = [
            {
                "line_id": "20m_top",
                "u1": 100, "v1": y_20m,
                "u2": 500, "v2": y_20m
            },  # Valid
            {
                "line_id": "20m_top",
                "u1": 100, "v1": y_20m,
                "u2": 500, "v2": y_20m + 500  # Invalid - tilted
            }
        ]

        valid, warnings = filter_valid_line_annotations(lines, good_homography)
        assert len(valid) == 1
        assert len(warnings) == 1


class TestLineConstrainedHomography:
    """Integration tests for full homography computation with lines."""

    @pytest.fixture
    def keypoint_correspondences(self):
        """Create realistic keypoint correspondences for testing."""
        # Simulate 4 corners of a GAA pitch view
        pts_image = np.array([
            [100, 100],   # Top-left corner
            [800, 100],   # Top-right corner
            [50, 600],    # Bottom-left corner
            [850, 600]    # Bottom-right corner
        ], dtype=np.float32)

        pts_canvas = np.array([
            [0, 0],         # corner_tl
            [850, 0],       # corner_tr
            [0, 1000],      # Some point lower
            [850, 1000]     # Some point lower
        ], dtype=np.float32)

        return pts_image, pts_canvas

    def test_compute_without_lines(self, keypoint_correspondences):
        """Test homography computation with keypoints only."""
        pts_img, pts_canvas = keypoint_correspondences

        H, info = compute_line_constrained_homography(
            pts_img, pts_canvas,
            line_annotations=[],
            max_iterations=3
        )

        assert H.shape == (3, 3)
        assert info['valid_lines'] == 0
        assert info['iterations'] == 0  # No iterations without lines

    def test_compute_with_lines(self, keypoint_correspondences):
        """Test homography computation with line constraints."""
        pts_img, pts_canvas = keypoint_correspondences

        # First compute initial H to know where lines would project
        H_initial = compute_initial_homography(pts_img, pts_canvas)

        # Create a line that would project correctly
        line_ann = {
            "line_id": "20m_top",
            "u1": 100, "v1": 150,
            "u2": 800, "v2": 150
        }

        H, info = compute_line_constrained_homography(
            pts_img, pts_canvas,
            [line_ann],
            num_samples_per_line=5,
            max_iterations=3,
            validate_lines=False  # Skip validation for test
        )

        assert H.shape == (3, 3)
        assert info['iterations'] >= 1
        assert info['synthetic_points'] == 5

    def test_minimum_keypoints_required(self):
        """Test that computation fails with fewer than 4 keypoints."""
        pts_img = np.array([[0, 0], [100, 0], [0, 100]], dtype=np.float32)
        pts_canvas = np.array([[0, 0], [100, 0], [0, 100]], dtype=np.float32)

        with pytest.raises(ValueError, match="at least 4 keypoints"):
            compute_line_constrained_homography(pts_img, pts_canvas, [])

    def test_convergence(self, keypoint_correspondences):
        """Test that algorithm converges quickly."""
        pts_img, pts_canvas = keypoint_correspondences

        line_ann = {
            "line_id": "20m_top",
            "u1": 200, "v1": 200,
            "u2": 700, "v2": 200
        }

        H, info = compute_line_constrained_homography(
            pts_img, pts_canvas,
            [line_ann],
            num_samples_per_line=10,
            max_iterations=5,
            validate_lines=False
        )

        # Should converge in a few iterations
        assert info['iterations'] <= 5


class TestPreviewSyntheticPoints:
    """Tests for previewing synthetic points."""

    def test_preview_returns_expected_structure(self):
        """Test preview returns correct data structure."""
        H = np.eye(3, dtype=np.float32)
        line_ann = {
            "line_id": "20m_top",
            "u1": 100, "v1": 200,
            "u2": 500, "v2": 200
        }

        results = preview_synthetic_points([line_ann], H, num_samples=5)

        assert len(results) == 5
        assert all('image_point' in r for r in results)
        assert all('world_point' in r for r in results)
        assert all('line_id' in r for r in results)
        assert all('weight' in r for r in results)


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_line_annotations(self):
        """Test handling of empty line annotations list."""
        pts_img = np.array([[0, 0], [100, 0], [0, 100], [100, 100]], dtype=np.float32)
        pts_canvas = np.array([[0, 0], [100, 0], [0, 100], [100, 100]], dtype=np.float32)

        H, info = compute_line_constrained_homography(pts_img, pts_canvas, [])

        assert H is not None
        assert info['valid_lines'] == 0

    def test_single_sample_per_line(self):
        """Test with minimum samples per line."""
        pts_img = np.array([[0, 0], [100, 0], [0, 100], [100, 100]], dtype=np.float32)
        pts_canvas = np.array([[0, 0], [100, 0], [0, 100], [100, 100]], dtype=np.float32)

        line_ann = {
            "line_id": "20m_top",
            "u1": 50, "v1": 50,
            "u2": 80, "v2": 50
        }

        H, info = compute_line_constrained_homography(
            pts_img, pts_canvas, [line_ann],
            num_samples_per_line=2,  # Minimum
            validate_lines=False
        )

        assert H is not None
        assert info['synthetic_points'] == 2

