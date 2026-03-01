# GAA Video Analysis Frontend

Frontend for the GAA Video Analysis System, deployed on Vercel.

## Features

- Upload MP4 videos
- Add pitch annotations for anchor frames
- Process videos through the backend pipeline
- Visualize player trajectories on a 850x1450 pitch canvas
- Navigate through frames to see player positions

## Setup

1. Install dependencies:
```bash
npm install
```

2. Set environment variable:
```bash
NEXT_PUBLIC_API_URL=https://your-render-backend-url.onrender.com
```

3. Run development server:
```bash
npm run dev
```

4. Build for production:
```bash
npm run build
npm start
```

## Deployment on Vercel

1. Connect your GitHub repository to Vercel
2. Set environment variable `NEXT_PUBLIC_API_URL` to your Render backend URL
3. Deploy

## Usage

1. Upload an MP4 video file
2. Add pitch annotations for anchor frames (at least 4 points per frame)
3. Click "Process Video" to start processing
4. View player trajectories on the pitch canvas
5. Use Previous/Next Frame buttons to navigate through frames
