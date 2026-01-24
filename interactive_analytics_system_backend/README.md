# GAA Video Analysis API

Production-ready FastAPI backend for GAA video analysis with player tracking and trajectory interpolation.

## Features

- **Video Upload & Processing** - Upload MP4 videos and extract metadata
- **Player Detection** - YOLOv8 + ByteTrack for player tracking
- **Pitch Calibration** - Homography-based coordinate mapping
- **Trajectory Interpolation** - Linear interpolation between anchor frames
- **RESTful API** - Clean JSON API with OpenAPI documentation

## Quick Start

### Local Development

```bash
# Clone and navigate to backend
cd interactive_analytics_system_backend

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt  # For testing

# Run server
uvicorn app:app --reload --port 8000
```

### Docker

```bash
# Build image
docker build -t gaa-video-api .

# Run container
docker run -p 8000:8000 \
  -e ALLOWED_ORIGINS=https://your-frontend.com \
  -v $(pwd)/data:/app/data \
  gaa-video-api
```

### Deploy to Render

1. Connect your GitHub repository to Render
2. Create a new Web Service pointing to `interactive_analytics_system_backend/`
3. Render will auto-detect `render.yaml` configuration
4. Set `ALLOWED_ORIGINS` in the Render dashboard to your frontend URL

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| POST | `/videos` | Upload video |
| POST | `/videos/{id}/track` | Run player tracking |
| POST | `/videos/{id}/homographies` | Compute homographies |
| POST | `/videos/{id}/map_players` | Map players to pitch |
| POST | `/videos/{id}/interpolate` | Interpolate trajectories |
| GET | `/videos/{id}/players` | Get player positions |
| POST | `/process-video` | Full pipeline (single request) |

### Interactive Documentation

- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Configuration

Environment variables (with defaults):

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | `8000` | Server port |
| `ALLOWED_ORIGINS` | `*` | CORS origins (comma-separated) |
| `MAX_VIDEO_SIZE_MB` | `500` | Max upload size |
| `DATA_DIR` | `data` | Data storage directory |
| `YOLO_MODEL_PATH` | `models/best.pt` | Path to YOLO model |

## Project Structure

```
interactive_analytics_system_backend/
├── app.py                 # FastAPI application
├── main.py                # Uvicorn entry point
├── pipeline/              # Core processing modules
│   ├── __init__.py
│   ├── config.py          # Configuration constants
│   ├── detect.py          # YOLO + ByteTrack detection
│   ├── gaa_pitch_config.py # GAA pitch vertices
│   ├── homography.py      # Homography computation
│   ├── map_players.py     # Pixel to pitch mapping
│   ├── schemas.py         # Pydantic models
│   ├── trajectories.py    # Trajectory interpolation
│   └── video.py           # Video utilities
├── models/                # YOLO model weights
│   └── best.pt
├── tests/                 # Test suite
├── requirements.txt       # Production dependencies
├── requirements-dev.txt   # Development dependencies
├── Dockerfile             # Container configuration
├── render.yaml            # Render deployment config
└── .env.example           # Environment template
```

## Testing

```bash
# Run all tests
pytest -v

# Run with coverage
pytest --cov=pipeline --cov-report=html
```

## Data Contract

The API uses these core data structures:

1. **Detection** - YOLO tracking output
   ```json
   {"frame_idx": 0, "track_id": 1, "x1": 100, "y1": 100, "x2": 150, "y2": 200, "confidence": 0.9}
   ```

2. **PitchAnnotation** - Keypoint annotations
   ```json
   {"frame_idx": 0, "points": [{"pitch_id": "corner_tl", "x_img": 100, "y_img": 50}]}
   ```

3. **PlayerPitchPosition** - Mapped positions
   ```json
   {"frame_idx": 0, "track_id": 1, "x_pitch": 425.0, "y_pitch": 725.0, "source": "homography"}
   ```

## License

MIT

