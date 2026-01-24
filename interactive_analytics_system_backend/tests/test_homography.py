from pipeline.homography import compute_homographies_from_annotations, map_pixel_to_distorted_pitch
import numpy as np


def test_compute_homographies_with_valid_annotations(sample_annotations):
    annotations_dict = {ann.frame_idx: ann.points for ann in sample_annotations}
    homogs = compute_homographies_from_annotations(annotations_dict)
    assert isinstance(homogs, dict)
    assert 0 in homogs
    H = homogs[0]
    assert H.shape == (3, 3)


def test_map_pixel_identity_homography(sample_homography):
    H = sample_homography[0]
    x, y = map_pixel_to_distorted_pitch(100.0, 50.0, H, out_w=400, out_h=800, k1=0.0)
    # With identity H and k1=0, expect same x,y
    assert abs(x - 100.0) < 1e-6
    assert abs(y - 50.0) < 1e-6
