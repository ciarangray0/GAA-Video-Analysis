# GPU Inference Setup

This directory contains the code for running YOLO + ByteTrack tracking on serverless GPUs.

## Why Remote GPU?

Running YOLO tracking on Render's CPU instances:
- Takes too long (can timeout)
- Is expensive for compute time
- Doesn't scale well

Using serverless GPUs:
- Fast inference (~10-30 seconds for a 1-minute video)
- Pay only for compute time used
- Automatic scaling

## Option 1: Modal (Recommended)

Modal is the easiest to set up and has a generous free tier.

### Setup Steps

1. **Install Modal CLI:**
   ```bash
   pip install modal
   ```

2. **Authenticate with Modal:**
   ```bash
   modal token new
   ```
   This will open a browser to authenticate.

3. **Deploy the YOLO service:**
   ```bash
   cd interactive_analytics_system_backend
   modal deploy gpu_inference/modal_yolo.py
   ```

4. **Get your endpoint URL:**
   After deployment, Modal will print something like:
   ```
   Created web endpoint: https://your-username--gaa-yolo-tracking-yolotracker-track-video-endpoint.modal.run
   ```

5. **Configure your Render backend:**
   Add these environment variables to your Render service:
   ```
   GPU_PROVIDER=modal
   GPU_ENDPOINT_URL=https://your-username--gaa-yolo-tracking-yolotracker-track-video-endpoint.modal.run
   ```

### Testing Locally

```bash
# Test with a video file
modal run gpu_inference/modal_yolo.py -- path/to/video.mp4
```

### Costs

- Modal free tier: $30/month of compute credits
- T4 GPU: ~$0.000164/second (~$0.59/hour)
- A 1-minute video typically takes 10-30 seconds = ~$0.005

## Option 2: RunPod Serverless

For higher volume or if you need more control.

### Setup Steps

1. **Create RunPod account:** https://runpod.io

2. **Create a Serverless Endpoint:**
   - Go to Serverless > Create Endpoint
   - Use Docker image or create custom template
   - Select GPU (T4 or RTX 3090 recommended)

3. **Configure environment:**
   ```
   GPU_PROVIDER=runpod
   GPU_ENDPOINT_URL=https://api.runpod.ai/v2/your-endpoint-id/run
   GPU_API_KEY=your-runpod-api-key
   ```

## Option 3: Google Colab (Free, Manual)

If you want a completely free option but don't mind manual steps:

1. Open the Colab notebook (link TBD)
2. Upload your video
3. Run YOLO tracking
4. Download the detections JSON
5. Upload to your app

## Environment Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `GPU_PROVIDER` | Which provider to use | `modal`, `runpod`, `local` |
| `GPU_ENDPOINT_URL` | The endpoint URL | `https://...modal.run` |
| `GPU_API_KEY` | API key (if required) | `rp_xxxxxxxx` |

## Local Fallback

If you set `GPU_PROVIDER=local`, the system will fall back to running YOLO on the local CPU. This requires uncommenting the torch dependencies in `requirements.txt` and will be slow, but works for testing.

## Architecture

```
┌─────────────────────┐      ┌─────────────────────┐
│   Render Backend    │      │   Modal GPU         │
│   (FastAPI)         │      │   (T4 GPU)          │
│                     │      │                     │
│  1. Upload video    │      │                     │
│  2. Call GPU API ───────>  │  3. Run YOLO        │
│                     │  <───── 4. Return detections│
│  5. Compute homography     │                     │
│  6. Map players     │      │                     │
│  7. Return results  │      │                     │
└─────────────────────┘      └─────────────────────┘
```

## Troubleshooting

### "Modal endpoint not responding"
- Check Modal dashboard for logs
- Ensure the endpoint is deployed: `modal deploy gpu_inference/modal_yolo.py`
- Cold starts can take 10-30 seconds on first request

### "Timeout waiting for response"
- Large videos (>5 minutes) may take longer
- Consider increasing timeout in GPU client
- Check Modal logs for errors

### "Out of memory"
- Very long videos may exceed GPU memory
- Consider processing in chunks
- Use a GPU with more VRAM (A10G instead of T4)

