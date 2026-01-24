import json
from io import BytesIO


def test_upload_video_and_track(client, sample_video_metadata, monkeypatch, sample_detections):
    # Monkeypatch run_tracking to return sample detections
    def fake_run_tracking(path):
        return sample_detections
    monkeypatch.setattr("pipeline.detect.run_tracking", fake_run_tracking)
    # `app.run_tracking` may not exist because the import is lazy; allow setting it without raising
    monkeypatch.setattr("app.run_tracking", fake_run_tracking, raising=False)

    # Upload a fake video file
    fake_file = BytesIO(b"fake mp4 data")
    response = client.post(
        "/videos",
        files={"file": ("test.mp4", fake_file, "video/mp4")}
    )
    assert response.status_code == 200
    body = response.json()
    video_id = body["video_id"]

    # Run tracking
    resp2 = client.post(f"/videos/{video_id}/track")
    assert resp2.status_code == 200
    body2 = resp2.json()
    assert body2["frames_processed"] >= 1
    assert body2["tracks"] >= 1


def test_homographies_endpoint(client, sample_video_metadata, sample_annotations):
    # Upload a fake video first
    fake_file = BytesIO(b"fake mp4 data")
    r = client.post("/videos", files={"file": ("v.mp4", fake_file, "video/mp4")})
    assert r.status_code == 200
    vid = r.json()["video_id"]

    # Compute homographies with good annotations
    payload = [a.dict() for a in sample_annotations]
    resp = client.post(f"/videos/{vid}/homographies", json=payload)
    assert resp.status_code == 200
    frames = resp.json()["frames"]
    assert 0 in frames


def test_homographies_bad_annotations(client, sample_video_metadata, bad_annotations):
    fake_file = BytesIO(b"fake mp4 data")
    r = client.post("/videos", files={"file": ("v.mp4", fake_file, "video/mp4")})
    vid = r.json()["video_id"]

    payload = [a.dict() for a in bad_annotations]
    resp = client.post(f"/videos/{vid}/homographies", json=payload)
    assert resp.status_code == 400


def test_map_players_and_interpolate_full_flow(client, monkeypatch, sample_video_metadata, sample_detections, sample_annotations, sample_homography, sample_positions):
    # Patch run_tracking
    monkeypatch.setattr("pipeline.detect.run_tracking", lambda path: sample_detections)
    # Upload video
    fake_file = BytesIO(b"fake mp4 data")
    r = client.post("/videos", files={"file": ("v.mp4", fake_file, "video/mp4")})
    vid = r.json()["video_id"]

    # Run tracking
    client.post(f"/videos/{vid}/track")

    # Compute homographies (patch compute function to return our sample homography)
    monkeypatch.setattr("pipeline.homography.compute_homographies_from_annotations", lambda ann: sample_homography)
    payload = [a.dict() for a in sample_annotations]
    client.post(f"/videos/{vid}/homographies", json=payload)

    # Map players
    resp = client.post(f"/videos/{vid}/map_players")
    assert resp.status_code == 200
    positions = resp.json()
    assert len(positions) >= 1

    # Patch interpolate to return a small interpolated list (simulate behavior)
    def fake_interpolate(positions, start, end):
        # Return one interpolated position for frame 1
        return [
            {
                "frame_idx": 1,
                "track_id": positions[0]["track_id"] if isinstance(positions[0], dict) else positions[0].track_id,
                "x_pitch": 123.0,
                "y_pitch": 456.0,
                "source": "interpolated"
            }
        ]

    monkeypatch.setattr("pipeline.trajectories.interpolate_trajectories", fake_interpolate)

    resp2 = client.post(f"/videos/{vid}/interpolate?start_frame=0&end_frame=5")
    assert resp2.status_code == 200
    body = resp2.json()
    assert body["method"] == "linear"


def test_process_video_full_pipeline(client, monkeypatch, sample_annotations, sample_homography, sample_detections, sample_positions, sample_video_metadata):
    # Patch heavy functions
    monkeypatch.setattr("pipeline.detect.run_tracking", lambda path: sample_detections)
    monkeypatch.setattr("pipeline.homography.compute_homographies_from_annotations", lambda ann: sample_homography)
    monkeypatch.setattr("pipeline.map_players.map_players_to_pitch", lambda dets, homogs: sample_positions)
    monkeypatch.setattr("pipeline.trajectories.interpolate_trajectories", lambda pos, s, e: [])

    annotations_json = json.dumps([a.model_dump() for a in sample_annotations])
    fake_file = BytesIO(b"fake mp4 data")
    resp = client.post(
        "/process-video",
        data={"annotations_json": annotations_json},
        files={"file": ("v.mp4", fake_file, "video/mp4")}
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "completed"
    assert "player_positions" in body
