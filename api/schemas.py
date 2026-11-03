"""
schemas.py — Pydantic Request/Response Models
==============================================
Type-safe API contracts for the RAG Intelligence Platform API.
"""

from pydantic import BaseModel, Field
from typing import Optional


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=1000, description="The question to answer")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of passages to retrieve")

    model_config = {"json_schema_extra": {"example": {"question": "Who discovered penicillin?", "top_k": 5}}}


class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: list[str]
    answerable: bool
    retrieval_scores: list[float]
    hallucination_verdict: str
    hallucination_composite: Optional[float]
    total_latency_ms: float
    stage_latencies: dict
    top_k: int


class EvaluationRequest(BaseModel):
    question: str
    generated_answer: str
    reference_answer: str
    context_passages: list[dict]


class EvaluationResponse(BaseModel):
    rouge_l: float
    bertscore_f1: Optional[float]
    faithfulness: float
    hallucinated: bool
    composite_score: float


class HealthResponse(BaseModel):
    status: str
    version: str
    model: str
    corpus_size: int
    index_loaded: bool
