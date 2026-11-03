"""
rag_pipeline.py — End-to-End RAG Orchestrator
===============================================
Production-grade Retrieval-Augmented Generation pipeline that
connects the Embedder, VectorStore, Retriever, and Generator
into a single, observable query interface.

Author : Avijeet Tiwari | Google Data Scientist Portfolio
Dataset: SQuAD 2.0 / Wikipedia passage corpus (4,039 chunks)
"""

import os
import sys
import time
import logging
from pathlib import Path
from typing import Optional

_pkg_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _pkg_root not in sys.path:
    sys.path.insert(0, _pkg_root)

from src.embedder import Embedder
from src.vector_store import FAISSVectorStore
from src.retriever import Retriever
from src.generator import GeminiGenerator
from src.hallucination_detector import HallucinationDetector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Default paths
BASE_DIR = Path(__file__).resolve().parent.parent
FAISS_INDEX_PATH = BASE_DIR / "models" / "faiss_index.index"
FAISS_META_PATH = BASE_DIR / "models" / "faiss_index_meta.parquet"


class RAGPipeline:
    """
    End-to-end RAG pipeline.

    Pipeline stages:
    1. Query Encoding  — embed user question (MiniLM-L6-v2)
    2. Retrieval       — FAISS top-k similarity search
    3. Generation      — Gemini 1.5 Flash grounded answer
    4. Hallucination   — NLI-based faithfulness check
    5. Response        — Structured output with metadata

    Usage:
        pipeline = RAGPipeline(api_key="AIza...")
        result = pipeline.query("Who discovered penicillin?")
    """

    def __init__(
        self,
        api_key: str,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        gemini_model: str = "gemini-3.5-flash",
        top_k: int = 5,
        faiss_index_path: Optional[str] = None,
        faiss_meta_path: Optional[str] = None,
        detect_hallucinations: bool = True,
    ):
        self.top_k = top_k
        self.detect_hallucinations = detect_hallucinations

        logger.info("Initialising RAG Pipeline components...")

        # 1. Embedder
        self.embedder = Embedder(model_name=model_name)

        # 2. Vector store + retriever
        idx_path = faiss_index_path or str(FAISS_INDEX_PATH)
        meta_path = faiss_meta_path or str(FAISS_META_PATH)
        self.vector_store = FAISSVectorStore(embedding_dim=self.embedder.get_embedding_dim())
        self.vector_store.load(idx_path, meta_path)
        self.retriever = Retriever(embedder=self.embedder, vector_store=self.vector_store)

        # 3. Generator
        self.generator = GeminiGenerator(api_key=api_key, model_name=gemini_model)

        # 4. Hallucination detector (no embedder for speed)
        self.hallucination_detector = HallucinationDetector(use_semantic=False)

        logger.info("RAG Pipeline ready.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def query(self, question: str, top_k: Optional[int] = None) -> dict:
        """
        Run full RAG pipeline for a single question.

        Returns:
            dict with: question, answer, sources, answerable,
                       retrieval_scores, hallucination_verdict,
                       total_latency_ms, stage_latencies
        """
        k = top_k or self.top_k
        latencies = {}

        # Stage 1: Retrieve
        t0 = time.perf_counter()
        retrieved = self.retriever.retrieve(question, top_k=k)
        latencies["retrieval_ms"] = round((time.perf_counter() - t0) * 1000, 2)

        # Stage 2: Generate
        t0 = time.perf_counter()
        gen_result = self.generator.generate(
            question=question,
            retrieved_passages=retrieved,
            max_context_passages=k,
        )
        latencies["generation_ms"] = round((time.perf_counter() - t0) * 1000, 2)

        # Stage 3: Hallucination check
        halluc_result = None
        if self.detect_hallucinations:
            t0 = time.perf_counter()
            halluc_result = self.hallucination_detector.detect(
                answer=gen_result["answer"],
                context_passages=retrieved,
            )
            latencies["hallucination_ms"] = round((time.perf_counter() - t0) * 1000, 2)

        total_latency = sum(latencies.values())

        return {
            "question": question,
            "answer": gen_result["answer"],
            "sources": gen_result["sources"],
            "answerable": gen_result["answerable"],
            "retrieval_scores": [r.get("score", 0.0) for r in retrieved],
            "hallucination_verdict": halluc_result.get("verdict") if halluc_result else "N/A",
            "hallucination_composite": halluc_result.get("composite_score") if halluc_result else None,
            "total_latency_ms": round(total_latency, 2),
            "stage_latencies": latencies,
            "top_k": k,
        }

    def batch_query(
        self,
        questions: list[str],
        top_k: Optional[int] = None,
        delay: float = 0.3,
    ) -> list[dict]:
        """Batch query with rate-limit throttling."""
        results = []
        for i, q in enumerate(questions):
            try:
                r = self.query(q, top_k=top_k)
                results.append(r)
                logger.info(f"  [{i+1}/{len(questions)}] latency={r['total_latency_ms']}ms | verdict={r['hallucination_verdict']}")
                if i < len(questions) - 1:
                    time.sleep(delay)
            except Exception as e:
                logger.error(f"Query failed [{i}]: {e}")
                results.append({"question": q, "error": str(e)})
        return results


if __name__ == "__main__":
    import os

    api_key = os.environ.get("GEMINI_API_KEY", "")

    pipeline = RAGPipeline(api_key=api_key)

    questions = [
        "Who discovered penicillin?",
        "What is the speed of light?",
        "Who wrote Hamlet?",
    ]

    for q in questions:
        result = pipeline.query(q)
        print(f"\nQ: {result['question']}")
        print(f"A: {result['answer'][:200]}")
        print(f"Verdict: {result['hallucination_verdict']} | Latency: {result['total_latency_ms']}ms")
