#!/bin/bash

# Start script for GAA Video Analysis System
# This script starts both the FastAPI backend and Streamlit frontend

echo "Starting GAA Video Analysis System..."
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Check if YOLO_MODEL_PATH is set
if [ -z "$YOLO_MODEL_PATH" ]; then
    echo "⚠️  Warning: YOLO_MODEL_PATH not set. Using default 'best.pt'"
    echo "   Set it with: export YOLO_MODEL_PATH='path/to/your/model.pt'"
fi

echo ""
echo "Starting FastAPI backend on http://localhost:8000"
echo "Starting Streamlit frontend on http://localhost:8501"
echo ""
echo "Press Ctrl+C to stop both services"
echo ""

# Start backend in background
uvicorn app:app --reload --port 8000 &
BACKEND_PID=$!

# Wait a moment for backend to start
sleep 2

# Start frontend
streamlit run streamlit_app.py

# Cleanup: kill backend when frontend stops
kill $BACKEND_PID 2>/dev/null
