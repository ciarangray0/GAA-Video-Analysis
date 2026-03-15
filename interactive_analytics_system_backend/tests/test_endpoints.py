from io import BytesIO


def test_upload_video_and_track(client, sample_video_metadata, monkeypatch, sample_detections):
    monkeypatch.setattr("pipeline.detect.run_tracking", lambda path: sample_detections)
    monkeypatch.setattr("app.run_tracking", lambda path: sample_detections, raising=False)

    fake_file = BytesIO(b"fake mp4 data")
    response = client.post("/videos", files={"file": ("test.mp4", fake_file, "video/mp4")})
    assert response.status_code == 200
    video_id = response.json()["video_id"]

    resp2 = client.post(f"/videos/{video_id}/track")
    assert resp2.status_code == 200
    body2 = resp2.json()
    assert body2["frames_processed"] >= 1
    assert body2["tracks"] >= 1


def test_homographies_v2_with_lines(client, sample_video_metadata, sample_anchor_frame_annotations):
    """Test the v2 homography endpoint with line constraints."""
    fake_file = BytesIO(b"fake mp4 data")
    r = client.post("/videos", files={"file": ("v.mp4", fake_file, "video/mp4")})
    assert r.status_code == 200
    vid = r.json()["video_id"]

    payload = [a.model_dump() for a in sample_anchor_frame_annotations]
    resp = client.post(f"/videos/{vid}/homographies/v2", json=payload)
    assert resp.status_code == 200
    result = resp.json()
    assert "frames" in result
    assert 0 in result["frames"]
    assert "info" in result


def test_homographies_v2_without_lines(client, sample_video_metadata, sample_anchor_frame_annotations_no_lines):
    """Test the v2 endpoint works without line constraints."""
    fake_file = BytesIO(b"fake mp4 data")
    r = client.post("/videos", files={"file": ("v.mp4", fake_file, "video/mp4")})
    vid = r.json()["video_id"]

    payload = [a.model_dump() for a in sample_anchor_frame_annotations_no_lines]
    resp = client.post(f"/videos/{vid}/homographies/v2", json=payload)
    assert resp.status_code == 200
    assert 0 in resp.json()["frames"]


def test_get_available_lines(client):
    """Test the endpoint that returns available line IDs."""
    resp = client.get("/line-constraints/available-lines")
    assert resp.status_code == 200
    result = resp.json()
    assert "lines" in result
    assert "20m_top" in result["lines"]
    assert "halfway" in result["lines"]


def test_homographies_v2_bad_annotations(client, sample_video_metadata):
    """Test that v2 endpoint returns 400 when fewer than 4 keypoints are provided."""
    fake_file = BytesIO(b"fake mp4 data")
    r = client.post("/videos", files={"file": ("v.mp4", fake_file, "video/mp4")})
    vid = r.json()["video_id"]

    payload = [{"frame_idx": 0, "points": [
        {"pitch_id": "corner_tl", "x_img": 0, "y_img": 0},
        {"pitch_id": "corner_tr", "x_img": 400, "y_img": 0},
    ], "lines": []}]
    resp = client.post(f"/videos/{vid}/homographies/v2", json=payload)
    assert resp.status_code == 400


def test_map_players_and_interpolate_full_flow(
    client, monkeypatch, sample_video_metadata, sample_detections,
    sample_anchor_frame_annotations, sample_homography, sample_positions
):
    monkeypatch.setattr("pipeline.detect.run_tracking", lambda path: sample_detections)

    fake_file = BytesIO(b"fake mp4 data")
    r = client.post("/videos", files={"file": ("v.mp4", fake_file, "video/mp4")})
    vid = r.json()["video_id"]

    client.post(f"/videos/{vid}/track")

    # Patch v2 pipeline functions
    monkeypatch.setattr(
        "pipeline.homography.compute_homographies_with_lines",
        lambda ann, **kw: (sample_homography, {0: {"valid_lines": 0}}),
    )
    monkeypatch.setattr(
        "pipeline.constrained_homography.build_constrained_per_frame_H",
        lambda path, anchors, **kw: (sample_homography, {}),
    )

    payload = [a.model_dump() for a in sample_anchor_frame_annotations]
    client.post(f"/videos/{vid}/homographies/v2", json=payload)

    resp = client.post(f"/videos/{vid}/map_players")
    assert resp.status_code == 200
    assert len(resp.json()) >= 1

    monkeypatch.setattr(
        "pipeline.trajectories.interpolate_trajectories",
        lambda positions, start, end: [],
    )
    resp2 = client.post(f"/videos/{vid}/interpolate?start_frame=0&end_frame=5")
    assert resp2.status_code == 200
    assert resp2.json()["method"] == "linear"


def test_get_frame_video_not_found(client):
    response = client.get("/videos/nonexistent-id/frame/0")
    assert response.status_code == 404


def test_get_frame_invalid_frame_index(client, sample_video_metadata, monkeypatch):
    monkeypatch.setattr("app.extract_frame", lambda path, idx: b"fake jpeg data")

    fake_file = BytesIO(b"fake mp4 data")
    response = client.post("/videos", files={"file": ("test.mp4", fake_file, "video/mp4")})
    video_id = response.json()["video_id"]

    resp = client.get(f"/videos/{video_id}/frame/100")
    assert resp.status_code == 400


def test_get_frame_success(client, sample_video_metadata, monkeypatch):
    fake_jpeg = b"\xff\xd8\xff\xe0fake jpeg data"
    monkeypatch.setattr("app.extract_frame", lambda path, idx: fake_jpeg)

    fake_file = BytesIO(b"fake mp4 data")
    response = client.post("/videos", files={"file": ("test.mp4", fake_file, "video/mp4")})
    video_id = response.json()["video_id"]

    resp = client.get(f"/videos/{video_id}/frame/0")
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "image/jpeg"


def test_track_video_not_found(client):
    response = client.post("/videos/nonexistent-id/track")
    assert response.status_code == 404
