"""FastAPI application exposing the Captain Voice Assistant pipeline.

Endpoints:
  GET  /                    -> serves the minimal web UI (static/index.html)
  POST /api/query           -> runs the full pipeline for a text query
  GET  /api/audio/{file}    -> serves a previously synthesized audio file
  GET  /api/languages       -> supported target languages (for the UI dropdown)
  GET  /api/health          -> basic readiness check
"""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.config import settings
from app.pipeline import CaptainAssistantPipeline
from app.translation.translator import LANGUAGE_NAMES

app = FastAPI(title="Captain Voice Assistant", version="1.0.0")

_pipeline: CaptainAssistantPipeline | None = None


def get_pipeline() -> CaptainAssistantPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = CaptainAssistantPipeline()
    return _pipeline


class QueryRequest(BaseModel):
    text: str = Field(..., min_length=1, description="The Captain's text command/question")
    target_lang: str | None = Field(None, description="ISO-ish language code, e.g. 'am', 'fr'")
    synthesize_audio: bool = True


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/languages")
def languages():
    return {"default": settings.default_target_lang, "options": LANGUAGE_NAMES}


@app.post("/api/query")
def query(req: QueryRequest):
    pipeline = get_pipeline()
    try:
        result = pipeline.run(
            query=req.text,
            target_lang=req.target_lang,
            synthesize_audio=req.synthesize_audio,
        )
    except RuntimeError as exc:
        # e.g. missing ANTHROPIC_API_KEY -- surface as a clean 400 instead of a 500 traceback
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return result.to_api_dict()


@app.get("/api/audio/{filename}")
def get_audio(filename: str):
    if "/" in filename or "\\" in filename or ".." in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")
    path = (settings.audio_output_dir / filename).resolve()
    if not path.is_file() or path.parent.resolve() != settings.audio_output_dir.resolve():
        raise HTTPException(status_code=404, detail="Audio file not found")
    media_type = "audio/mpeg" if path.suffix == ".mp3" else "audio/wav"
    return FileResponse(path, media_type=media_type)


static_dir = Path(__file__).resolve().parent.parent / "static"
if static_dir.exists():
    app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")
