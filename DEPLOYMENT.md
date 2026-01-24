# Deployment Guide

This guide explains how to deploy the GAA Video Analysis System to Render (backend) and Vercel (frontend).

## Project Structure

```
GAA-Video-Analysis/
├── interactive_analytics_system_backend/    # FastAPI backend (Render)
├── Interactive_analytics_system_frontend/   # Next.js frontend (Vercel)
└── pipeline_testing_and_research/          # Ignore this folder
```

## Backend Deployment (Render)

### Prerequisites
- GitHub repository connected
- Render account (free tier works)

### Steps

1. **Create a new Web Service on Render**
   - Connect your GitHub repository
   - Select the `interactive_analytics_system_backend` folder as the root directory

2. **Configure Build Settings**
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `uvicorn app:app --host 0.0.0.0 --port $PORT`

3. **Set Environment Variables**
   - `YOLO_MODEL_PATH`: (optional) Path to your YOLO model. Defaults to `yolov8n.pt` which will be downloaded automatically.
   - `PORT`: Automatically set by Render

4. **Deploy**
   - Click "Create Web Service"
   - Wait for deployment to complete
   - Note your backend URL (e.g., `https://your-app.onrender.com`)

### Notes
- Render free tier has limitations: services spin down after 15 minutes of inactivity
- First request after spin-down may take 30-60 seconds
- CPU-only processing (no GPU) - processing may be slower than local GPU

## Frontend Deployment (Vercel)

### Prerequisites
- GitHub repository connected
- Vercel account (free tier works)
- Backend URL from Render

### Steps

1. **Import Project on Vercel**
   - Connect your GitHub repository
   - Select the `Interactive_analytics_system_frontend` folder as the root directory
   - Framework Preset: Next.js (auto-detected)

2. **Set Environment Variables**
   - `NEXT_PUBLIC_API_URL`: Your Render backend URL (e.g., `https://your-app.onrender.com`)

3. **Deploy**
   - Click "Deploy"
   - Wait for deployment to complete
   - Your frontend will be available at `https://your-app.vercel.app`

### Notes
- Vercel automatically builds and deploys on every push to main branch
- Environment variables can be updated in Vercel dashboard

## Testing Locally

### Backend
```bash
cd interactive_analytics_system_backend
pip install -r requirements.txt
uvicorn app:app --reload --port 8000
```

### Frontend
```bash
cd Interactive_analytics_system_frontend
npm install
NEXT_PUBLIC_API_URL=http://localhost:8000 npm run dev
```

## API Usage

### POST /process-video

**Endpoint:** `https://your-backend.onrender.com/process-video`

**Request:**
- Content-Type: `multipart/form-data`
- `file`: MP4 video file
- `annotations_json`: JSON string of pitch annotations

**Example using curl:**
```bash
curl -X POST https://your-backend.onrender.com/process-video \
  -F "file=@video.mp4" \
  -F 'annotations_json=[{"frame_idx":0,"points":[{"pitch_id":"corner_tl","x_img":100,"y_img":50},{"pitch_id":"corner_tr","x_img":900,"y_img":50},{"pitch_id":"corner_bl","x_img":100,"y_img":1000},{"pitch_id":"corner_br","x_img":900,"y_img":1000}]}]'
```

**Response:**
```json
{
  "video_id": "uuid",
  "status": "completed",
  "player_positions": [
    {
      "frame_idx": 0,
      "track_id": 1,
      "x_pitch": 425.0,
      "y_pitch": 725.0,
      "source": "homography"
    }
  ]
}
```

## Troubleshooting

### Backend Issues

**Service won't start:**
- Check build logs in Render dashboard
- Ensure `requirements.txt` is correct
- Verify Python version (Render uses Python 3.11 by default)

**Out of memory:**
- Render free tier has 512MB RAM limit
- Consider processing shorter videos
- Reduce video resolution if possible

**Slow processing:**
- CPU-only processing is slower than GPU
- Consider upgrading to paid tier for better performance

### Frontend Issues

**Can't connect to backend:**
- Verify `NEXT_PUBLIC_API_URL` is set correctly
- Check CORS settings (backend allows all origins)
- Ensure backend is running and accessible

**Build fails:**
- Check Vercel build logs
- Ensure all dependencies are in `package.json`
- Verify Node.js version (Vercel uses Node 18+)

## Cost Considerations

- **Render Free Tier:**
  - 750 hours/month
  - Services spin down after inactivity
  - 512MB RAM limit
  
- **Vercel Free Tier:**
  - Unlimited deployments
  - 100GB bandwidth/month
  - Serverless functions included

For production use, consider upgrading to paid tiers for better performance and reliability.
