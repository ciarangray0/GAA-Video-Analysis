# GAA Video Analysis System

A complete video analysis pipeline for tracking GAA (Gaelic Athletic Association) players, mapping them to pitch coordinates, and generating interpolated trajectories.

## Architecture

The system consists of two components:

1. **FastAPI Backend** (`app.py`) - REST API that handles video processing, tracking, homography computation, and trajectory interpolation
2. **Streamlit Frontend** (`streamlit_app.py`) - Interactive web interface for uploading videos and running the analysis pipeline

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set Environment Variables

```bash
export YOLO_MODEL_PATH="path/to/your/best.pt"  # Path to your trained YOLO model
```

If you don't have a model yet, you can use a pre-trained YOLOv8 model (e.g., `yolov8n.pt`) for testing, but it won't be trained on GAA players.

### 3. Run the Backend

In one terminal:

```bash
uvicorn app:app --reload --port 8000
```

The API will be available at:
- API: http://localhost:8000
- API Docs: http://localhost:8000/docs

### 4. Run the Frontend

In another terminal:

```bash
streamlit run streamlit_app.py
```

The frontend will open in your browser at http://localhost:8501

## Usage

### Pipeline Steps

1. **Upload Video** - Upload a video file (MP4, AVI, MOV, etc.)
2. **Run Tracking** - Execute YOLO + ByteTrack to detect and track players
3. **Annotate Pitch** - Mark pitch keypoints on selected frames for homography calibration
4. **Compute Homographies** - Calculate homography matrices for annotated frames
5. **Map Players** - Transform player detections from image space to pitch coordinates
6. **Interpolate** - Generate dense trajectories by interpolating between anchor frames
7. **View Results** - Browse and download player positions

### API Endpoints

- `POST /videos` - Upload video
- `POST /videos/{id}/track` - Run tracking
- `POST /videos/{id}/homographies` - Compute homographies from annotations
- `POST /videos/{id}/map_players` - Map players to pitch
- `POST /videos/{id}/interpolate?start_frame=X&end_frame=Y` - Interpolate trajectories
- `GET /videos/{id}/players` - Get all player positions

## Data Storage

All data is stored in the `data/` directory:
- `data/videos/` - Uploaded video files
- `data/tracks/` - Tracking results (JSON)
- `data/annotations/` - Homography matrices (JSON)

## Configuration

Pitch and processing parameters can be adjusted in `pipeline/config.py`:
- `PITCH_W`, `PITCH_H` - Pitch dimensions in meters
- `OUT_W`, `OUT_H` - Output canvas dimensions
- `K1` - Radial distortion coefficient
- `DEFAULT_CONF` - Default detection confidence threshold

## Notes

- The tracking step can take several minutes depending on video length
- You need at least 4 pitch point annotations per frame to compute a homography
- Interpolation requires at least 2 anchor frames with player positions
- The system uses bounding box bottom-center for player position mapping

## Troubleshooting

**Backend won't start:**
- Check that port 8000 is not in use
- Verify all dependencies are installed

**Streamlit can't connect to backend:**
- Ensure the FastAPI server is running on port 8000
- Check the `API_URL` in `streamlit_app.py` matches your backend URL

**Tracking fails:**
- Verify `YOLO_MODEL_PATH` points to a valid model file
- Check that the video file is valid and readable

**Homography computation fails:**
- Ensure you have at least 4 annotated points per frame
- Verify pitch point IDs match those in `gaa_pitch_config.py`
