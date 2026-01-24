from pipeline.trajectories import interpolate_trajectories
from pipeline.schemas import PlayerPitchPosition


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

