from pipeline.trajectories import interpolate_trajectories
from pipeline.schemas import PlayerPitchPosition
from pipeline.config import OUT_W, OUT_H


def test_interpolate_linear_two_anchors():
    sparse = [
        PlayerPitchPosition(frame_idx=0, track_id=1, x_pitch=0.0, y_pitch=0.0, source="homography"),
        PlayerPitchPosition(frame_idx=10, track_id=1, x_pitch=10.0, y_pitch=10.0, source="homography")
    ]
    interpolated = interpolate_trajectories(sparse, 0, 10)
    # Should contain frames 0..10 inclusive
    frames = sorted(set(p.frame_idx for p in interpolated))
    assert frames[0] == 0
    assert frames[-1] == 10
    # Check middle frame ~5
    mid = next(p for p in interpolated if p.frame_idx == 5 and p.track_id == 1)
    assert abs(mid.x_pitch - 5.0) < 1e-6
    assert abs(mid.y_pitch - 5.0) < 1e-6
    # Anchors preserved
    start = next(p for p in interpolated if p.frame_idx == 0 and p.track_id == 1)
    assert start.source == "homography"
    end = next(p for p in interpolated if p.frame_idx == 10 and p.track_id == 1)
    assert end.source == "homography"


def test_interpolate_single_anchor_returns_anchor_only():
    sparse = [
        PlayerPitchPosition(frame_idx=4, track_id=2, x_pitch=50.0, y_pitch=60.0, source="homography")
    ]
    interpolated = interpolate_trajectories(sparse, 0, 10)
    # Only the anchor should be present
    assert len(interpolated) == 1
    p = interpolated[0]
    assert p.frame_idx == 4
    assert p.source == "homography"


def test_interpolate_cubic_spline_three_anchors():
    """Three anchor points triggers cubic spline; anchors are preserved exactly."""
    sparse = [
        PlayerPitchPosition(frame_idx=0,  track_id=1, x_pitch=0.0,   y_pitch=0.0,   source="homography"),
        PlayerPitchPosition(frame_idx=5,  track_id=1, x_pitch=50.0,  y_pitch=50.0,  source="homography"),
        PlayerPitchPosition(frame_idx=10, track_id=1, x_pitch=100.0, y_pitch=100.0, source="homography"),
    ]
    interpolated = interpolate_trajectories(sparse, 0, 10)

    frames = sorted(set(p.frame_idx for p in interpolated))
    assert frames[0] == 0
    assert frames[-1] == 10
    assert len(frames) == 11

    # Original anchor frames must be preserved
    for frame_idx in [0, 5, 10]:
        anchor = next(p for p in interpolated if p.frame_idx == frame_idx)
        assert anchor.source == "homography"


def test_interpolate_coordinates_clamped_to_pitch():
    """Interpolated positions must not exceed pitch canvas bounds."""
    # Place anchors near the edges to generate values that might overshoot
    sparse = [
        PlayerPitchPosition(frame_idx=0,  track_id=3, x_pitch=0.0,   y_pitch=0.0,   source="homography"),
        PlayerPitchPosition(frame_idx=5,  track_id=3, x_pitch=OUT_W, y_pitch=OUT_H,  source="homography"),
        PlayerPitchPosition(frame_idx=10, track_id=3, x_pitch=0.0,   y_pitch=0.0,   source="homography"),
    ]
    interpolated = interpolate_trajectories(sparse, 0, 10)

    for pos in interpolated:
        assert 0.0 <= pos.x_pitch <= OUT_W, f"x_pitch {pos.x_pitch} out of bounds"
        assert 0.0 <= pos.y_pitch <= OUT_H, f"y_pitch {pos.y_pitch} out of bounds"


def test_interpolate_savgol_applied_for_long_tracks():
    """Savitzky-Golay smoothing is applied for tracks with >15 data points."""
    # Create 3 anchors across 30 frames — the result will have 31 points
    sparse = [
        PlayerPitchPosition(frame_idx=0,  track_id=5, x_pitch=0.0,   y_pitch=0.0,   source="homography"),
        PlayerPitchPosition(frame_idx=15, track_id=5, x_pitch=425.0, y_pitch=700.0, source="homography"),
        PlayerPitchPosition(frame_idx=30, track_id=5, x_pitch=850.0, y_pitch=1400.0, source="homography"),
    ]
    interpolated = interpolate_trajectories(sparse, 0, 30)

    # 31 frames total
    assert len(interpolated) == 31

    # All values should still be in bounds after Savitzky-Golay
    for pos in interpolated:
        assert 0.0 <= pos.x_pitch <= OUT_W
        assert 0.0 <= pos.y_pitch <= OUT_H

