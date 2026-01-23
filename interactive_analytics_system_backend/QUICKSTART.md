# Quick Start Guide

## Option 1: Use the Start Script (Easiest)

```bash
./start.sh
```

This will:
- Create a virtual environment (if needed)
- Install all dependencies
- Start the FastAPI backend on port 8000
- Start the Streamlit frontend on port 8501

## Option 2: Manual Start

### Terminal 1 - Backend
```bash
# Install dependencies (first time only)
pip install -r requirements.txt

# Set model path (optional, defaults to 'best.pt')
export YOLO_MODEL_PATH="path/to/your/best.pt"

# Start backend
uvicorn app:app --reload --port 8000
```

### Terminal 2 - Frontend
```bash
# Start Streamlit
streamlit run streamlit_app.py
```

## Access the Application

- **Frontend**: http://localhost:8501
- **Backend API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs

## First Steps

1. Open http://localhost:8501 in your browser
2. Use the sidebar to navigate through the pipeline steps
3. Upload a video file (Step 1)
4. Run tracking (Step 2) - this may take several minutes
5. Annotate pitch points on key frames (Step 3)
6. Compute homographies (Step 4)
7. Map players to pitch (Step 5)
8. Interpolate trajectories (Step 6)
9. View results (Step 7)

## Troubleshooting

**Port already in use?**
- Change ports in the commands:
  - Backend: `uvicorn app:app --reload --port 8001`
  - Frontend: `streamlit run streamlit_app.py --server.port 8502`
  - Update `API_URL` in `streamlit_app.py` to match

**Model not found?**
- Set `YOLO_MODEL_PATH` environment variable
- Or place your model file as `best.pt` in the project root

**Dependencies issues?**
- Use a virtual environment: `python3 -m venv venv && source venv/bin/activate`
- Reinstall: `pip install -r requirements.txt --upgrade`
