"""Entry point for Render deployment."""
import os
import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        workers=1  # Single worker for free tier
    )
