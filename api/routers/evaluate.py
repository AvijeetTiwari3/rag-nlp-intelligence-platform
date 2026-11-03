"""evaluate.py — Evaluation Router"""
import os
import sys
from fastapi import APIRouter, HTTPException

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from api.schemas import EvaluationRequest, EvaluationResponse
from src.evaluator import RAGEvaluator

router = APIRouter()
_evaluator = RAGEvaluator(use_bertscore=False)


@router.post("/evaluate", response_model=EvaluationResponse)
async def evaluate_answer(request: EvaluationRequest):
    try:
        result = _evaluator.evaluate_single(
            question=request.question,
            generated_answer=request.generated_answer,
            reference_answer=request.reference_answer,
            context_passages=request.context_passages,
        )
        return EvaluationResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
