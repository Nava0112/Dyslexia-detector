import uuid
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.config import (
    ALLOWED_CONTENT_TYPES,
    ARTIFACT_DIR,
    CHECKPOINT_PATH,
    MAX_UPLOAD_BYTES,
)
from backend.pipeline import AnalysisPipeline
from backend.guidance import GuidanceError, create_guidance
from backend.schemas import AnalysisResponse, GuidanceRequest


app = FastAPI(
    title="Dyslexia Handwriting Screening API",
    version="1.0.0",
    description="Local handwriting screening API. This is not a clinical diagnostic system.",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
pipeline: AnalysisPipeline | None = None
FRONTEND_DIR = Path(__file__).resolve().parent / "frontend"

app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

@app.get("/")
def home():
    return FileResponse(FRONTEND_DIR / "index.html")

@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/ready")
def ready():
    if not CHECKPOINT_PATH.is_file():
        raise HTTPException(
            status_code=503,
            detail=f"Model checkpoint not found: {CHECKPOINT_PATH}",
        )
    return {"status": "ready"}


@app.post("/analyze", response_model=AnalysisResponse)
async def analyze(file: UploadFile = File(...)):
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=415,
            detail="Only JPG, PNG, and WebP images are accepted.",
        )
    payload = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(payload) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail="Image exceeds the 10 MB upload limit.",
        )
    image = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(
            status_code=422,
            detail="The uploaded file is not a readable image.",
        )

    global pipeline
    if pipeline is None:
        try:
            pipeline = AnalysisPipeline(CHECKPOINT_PATH)
        except (FileNotFoundError, RuntimeError) as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
    try:
        return pipeline.analyze(image, uuid.uuid4().hex)
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/guidance")
async def guidance(request: GuidanceRequest):
    counts = request.class_counts
    required_classes = {"Normal", "Corrected", "Reversal"}
    if set(counts) != required_classes or any(value < 0 for value in counts.values()):
        raise HTTPException(
            status_code=422,
            detail="class_counts must contain exactly Normal, Corrected, and Reversal.",
        )
    try:
        content = create_guidance(request.model_dump())
    except GuidanceError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {"content": content}


@app.get("/artifacts/{job_id}/{filename}")
def artifact(job_id: str, filename: str):
    if len(job_id) != 32 or any(character not in "0123456789abcdef" for character in job_id):
        raise HTTPException(status_code=404, detail="Artifact not found.")
    if filename not in {"annotated.png", "result.json"}:
        raise HTTPException(status_code=404, detail="Artifact not found.")
    path = ARTIFACT_DIR / job_id / filename
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Artifact not found.")
    return FileResponse(path)