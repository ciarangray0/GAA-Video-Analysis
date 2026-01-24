from pipeline.map_players import map_players_to_pitch
from pipeline.schemas import Detection


def test_map_players_bottom_center(sample_detections, sample_homography):
    # Use the sample detections and homography (identity)
    positions = map_players_to_pitch(sample_detections, sample_homography, out_w=400, out_h=800, k1=0.0)
    # Should map entries only for frames that have homography (frame 0 for our sample_homography)
    assert all(p.source == "homography" for p in positions)
    # For frame 0, there are two detections (track 1 and 2)
    frame0_positions = [p for p in positions if p.frame_idx == 0]
    assert len(frame0_positions) == 2
    # Check bottom-center calculation for first detection
    first_det = next(d for d in sample_detections if d.frame_idx == 0 and d.track_id == 1)
    expected_x = (first_det.x1 + first_det.x2) / 2
    expected_y = first_det.y2
    p = next(p for p in frame0_positions if p.track_id == 1)
    assert abs(p.x_pitch - expected_x) < 1e-6
    assert abs(p.y_pitch - expected_y) < 1e-6

