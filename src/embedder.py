"""
Sentence-BERT Embedding Engine & Multi-Model Benchmarking Suite
Supports all-MiniLM-L6-v2, all-mpnet-base-v2, and intfloat/e5-small-v2.
"""

import time
import os
import sys
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Union
import numpy as np
from sentence_transformers import SentenceTransformer

# Priority sys.path insertion
_pkg_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _pkg_root not in sys.path:
    sys.path.insert(0, _pkg_root)


@dataclass
class EmbeddingBenchmarkResult:
    model_name: str
    embedding_dimension: int
    total_chunks_encoded: int
    total_encoding_time_sec: float
    throughput_chunks_per_sec: float
    avg_latency_ms_per_query: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class EmbeddingEngine:
    """Production Sentence-BERT embedding engine with L2 normalization."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        # Normalize model identifier
        if "/" not in model_name and not model_name.startswith("sentence-transformers"):
            full_model_name = f"sentence-transformers/{model_name}"
        else:
            full_model_name = model_name

        self.model_name = full_model_name
        self.model = SentenceTransformer(full_model_name)
        if hasattr(self.model, "get_embedding_dimension"):
            self.embedding_dim = self.model.get_embedding_dimension()
        elif hasattr(self.model, "get_sentence_embedding_dimension"):
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
        else:
            self.embedding_dim = 384

    def get_embedding_dim(self) -> int:
        return self.embedding_dim

    def encode(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 64,
        show_progress: bool = False
    ) -> np.ndarray:
        """Universal encode method for single strings or lists of texts."""
        if isinstance(texts, str):
            texts = [texts]
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=True,
            convert_to_numpy=True
        )
        return embeddings.astype(np.float32)

    def encode_corpus(
        self,
        texts: List[str],
        batch_size: int = 64,
        show_progress: bool = True
    ) -> np.ndarray:
        """Batch encode documents into normalized L2 float32 vectors."""
        return self.encode(texts, batch_size=batch_size, show_progress=show_progress)

    def encode_query(self, query: str) -> np.ndarray:
        """Encode a single user query into a 1xD normalized float32 vector."""
        return self.encode(query, show_progress=False)


# Universal Alias
Embedder = EmbeddingEngine


def benchmark_embedding_models(
    sample_texts: List[str],
    sample_queries: List[str],
    model_names: Optional[List[str]] = None
) -> List[EmbeddingBenchmarkResult]:
    """Benchmark multiple embedding models on encoding throughput and query latency."""
    if model_names is None:
        model_names = [
            "all-MiniLM-L6-v2"
        ]

    results = []
    for m in model_names:
        print(f"Benchmarking embedding model: {m}...")
        engine = EmbeddingEngine(m)

        # 1. Corpus throughput test
        t0 = time.perf_counter()
        embs = engine.encode_corpus(sample_texts, batch_size=64, show_progress=False)
        t_enc = time.perf_counter() - t0

        # 2. Query latency test
        q_latencies = []
        for q in sample_queries[:50]:
            t_q0 = time.perf_counter()
            _ = engine.encode_query(q)
            q_latencies.append((time.perf_counter() - t_q0) * 1000.0)

        res = EmbeddingBenchmarkResult(
            model_name=m,
            embedding_dimension=engine.embedding_dim,
            total_chunks_encoded=len(sample_texts),
            total_encoding_time_sec=round(t_enc, 2),
            throughput_chunks_per_sec=round(len(sample_texts) / max(t_enc, 0.001), 1),
            avg_latency_ms_per_query=round(float(np.mean(q_latencies)), 2)
        )
        results.append(res)

    return results


if __name__ == "__main__":
    engine = EmbeddingEngine("all-MiniLM-L6-v2")
    embs = engine.encode(["Hello world", "RAG systems"])
    print(f"Encoded shape: {embs.shape}, Dim: {engine.get_embedding_dim()}")
