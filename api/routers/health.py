"""health.py — Health Check Router"""
from fastapi import APIRouter
from api.schemas import HealthResponse
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check():
    from api.app import app_state
    pipeline = app_state.get("pipeline")
    return HealthResponse(
        status="healthy" if pipeline else "degraded",
        version="1.0.0",
        model="gemini-1.5-flash",
        corpus_size=4039,
        index_loaded=pipeline is not None,
    )
