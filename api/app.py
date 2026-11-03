"""
app.py — FastAPI Application Entry Point
==========================================
Production-grade FastAPI app for the RAG NLP Intelligence Platform.
Includes: CORS, request logging, error handling, startup lifecycle.

Author : Avijeet Tiwari | Google Data Scientist Portfolio
"""

import os
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routers import health, query, evaluate

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

API_VERSION = "1.0.0"
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

# Shared state for pipeline (loaded once at startup)
app_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load RAG pipeline at startup, release at shutdown."""
    logger.info("Loading RAG Pipeline...")
    try:
        from src.rag_pipeline import RAGPipeline
        pipeline = RAGPipeline(api_key=GEMINI_API_KEY)
        app_state["pipeline"] = pipeline
        logger.info("RAG Pipeline loaded successfully.")
    except Exception as e:
        logger.error(f"Pipeline load failed: {e}")
        app_state["pipeline"] = None
        app_state["pipeline_error"] = str(e)

    yield

    logger.info("Shutting down RAG Pipeline.")
    app_state.clear()


app = FastAPI(
    title="RAG NLP Intelligence Platform API",
    description=(
        "Production-grade Retrieval-Augmented Generation API powered by "
        "SQuAD 2.0 Wikipedia corpus (4,039 chunks), MiniLM-L6-v2 embeddings, "
        "FAISS vector search, and Gemini 1.5 Flash generation."
    ),
    version=API_VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS — allow all origins for portfolio deployment
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers
app.include_router(health.router, prefix="/api/v1", tags=["Health"])
app.include_router(query.router, prefix="/api/v1", tags=["Query"])
app.include_router(evaluate.router, prefix="/api/v1", tags=["Evaluate"])


@app.get("/", tags=["Root"])
async def root():
    return {
        "project": "RAG NLP Intelligence Platform",
        "version": API_VERSION,
        "docs": "/docs",
        "health": "/api/v1/health",
        "author": "Avijeet Tiwari | Google Data Scientist Portfolio",
    }
