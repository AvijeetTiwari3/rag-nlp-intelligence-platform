"""query.py — RAG Query Router"""
import os
import sys
from fastapi import APIRouter, HTTPException

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from api.schemas import QueryRequest, QueryResponse

router = APIRouter()


@router.post("/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest):
    from api.app import app_state
    pipeline = app_state.get("pipeline")
    if pipeline is None:
        raise HTTPException(
            status_code=503,
            detail=f"RAG pipeline not available: {app_state.get('pipeline_error', 'unknown error')}",
        )
    try:
        result = pipeline.query(request.question, top_k=request.top_k)
        return QueryResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
