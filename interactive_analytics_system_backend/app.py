"""FastAPI application for video analysis pipeline."""
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from pipeline.persistence import ensure_dirs, restore_videos_from_disk
from store import store

from routes.videos import router as videos_router
from routes.detection import router as detection_router
from routes.homography import router as homography_router
from routes.mapping import router as mapping_router
from routes.classification import router as classification_router
from routes.kpi import router as kpi_router

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")


@asynccontextmanager
async def lifespan(app: FastAPI):
    ensure_dirs()
    videos = restore_videos_from_disk()
    store.videos.update(videos)
    if store.videos:
        logger.info(f"Restored {len(store.videos)} video(s) from disk on startup")
    yield


app = FastAPI(title="GAA Video Analysis API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(videos_router)
app.include_router(detection_router)
app.include_router(homography_router)
app.include_router(mapping_router)
app.include_router(classification_router)
app.include_router(kpi_router)


@app.get("/health")
async def health_check():
    return {"status": "ok"}