"""
benchmark.py — Multi-Model Embedding Benchmarking Suite
=========================================================
Systematic benchmarking of sentence embedding models for RAG retrieval.
Compares: MiniLM-L6, MiniLM-L12, MPNet-base, DistilRoBERTa
across Precision@K, Recall@K, MRR, NDCG@5, latency, and throughput.

Author : Avijeet Tiwari | Google Data Scientist Portfolio
Dataset: SQuAD 2.0 Golden Evaluation Set (200 samples)
"""

import os
import sys
import time
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

_pkg_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _pkg_root not in sys.path:
    sys.path.insert(0, _pkg_root)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent

CANDIDATE_MODELS = {
    "MiniLM-L6-v2": "sentence-transformers/all-MiniLM-L6-v2",
    "MiniLM-L12-v2": "sentence-transformers/all-MiniLM-L12-v2",
    "MPNet-base-v2": "sentence-transformers/all-mpnet-base-v2",
    "DistilRoBERTa": "sentence-transformers/all-distilroberta-v1",
}


class EmbeddingBenchmark:
    """
    Benchmarks multiple sentence embedding models on retrieval quality.

    Metrics:
    - Precision@1, Precision@5
    - Recall@5, Recall@10
    - MRR (Mean Reciprocal Rank)
    - NDCG@5
    - Encoding throughput (chunks/sec)
    - Query latency (ms)
    - Index size (MB)
    """

    def __init__(
        self,
        corpus_path: Optional[str] = None,
        golden_set_path: Optional[str] = None,
        n_eval_samples: int = 200,
    ):
        self.corpus_path = corpus_path or str(BASE_DIR / "data" / "processed" / "corpus_chunks.parquet")
        self.golden_set_path = golden_set_path or str(BASE_DIR / "data" / "processed" / "eval_golden_set.parquet")
        self.n_eval_samples = n_eval_samples
        self.results = []

    def run(self, models: Optional[dict] = None) -> pd.DataFrame:
        """Run benchmark across all candidate models."""
        from src.embedder import Embedder
        from src.vector_store import FAISSVectorStore

        models = models or CANDIDATE_MODELS
        corpus_df = pd.read_parquet(self.corpus_path)
        eval_df = pd.read_parquet(self.golden_set_path)
        eval_df = eval_df[eval_df["has_answer"]].head(self.n_eval_samples)

        logger.info(f"Corpus: {len(corpus_df)} chunks | Eval: {len(eval_df)} queries")

        for model_name, model_id in models.items():
            logger.info(f"\n{'='*50}")
            logger.info(f"Benchmarking: {model_name}")
            try:
                row = self._benchmark_model(model_name, model_id, corpus_df, eval_df)
                self.results.append(row)
            except Exception as e:
                logger.error(f"Failed {model_name}: {e}")

        results_df = pd.DataFrame(self.results).sort_values("mrr", ascending=False)
        return results_df

    def _benchmark_model(
        self,
        model_name: str,
        model_id: str,
        corpus_df: pd.DataFrame,
        eval_df: pd.DataFrame,
    ) -> dict:
        from src.embedder import Embedder
        from src.vector_store import FAISSVectorStore

        embedder = Embedder(model_name=model_id)
        dim = embedder.get_embedding_dim()

        # Encode corpus
        texts = corpus_df["text"].tolist()
        t_start = time.perf_counter()
        embeddings = embedder.encode(texts, batch_size=64, show_progress=False)
        encode_time = time.perf_counter() - t_start
        throughput = len(texts) / encode_time

        # Build index
        store = FAISSVectorStore(embedding_dim=dim)
        meta = corpus_df[["chunk_id", "title", "text"]].to_dict("records")
        store.add(embeddings, meta)

        # Index size (approx)
        index_size_mb = (len(texts) * dim * 4) / (1024 ** 2)

        # Evaluate retrieval
        precision_1_scores = []
        precision_5_scores = []
        recall_5_scores = []
        recall_10_scores = []
        mrr_scores = []
        ndcg_5_scores = []
        query_latencies = []

        for _, row in eval_df.iterrows():
            question = row["question"]
            context_id = row.get("context_id", "")

            t0 = time.perf_counter()
            q_emb = embedder.encode([question])[0]
            results = store.search(q_emb.reshape(1, -1), top_k=10)
            query_latencies.append((time.perf_counter() - t0) * 1000)

            retrieved_ids = [r.get("chunk_id", "") for r in results]
            hit_flags = [1 if cid == context_id else 0 for cid in retrieved_ids]

            precision_1_scores.append(hit_flags[0] if hit_flags else 0)
            precision_5_scores.append(sum(hit_flags[:5]) / 5)
            recall_5_scores.append(1.0 if any(hit_flags[:5]) else 0.0)
            recall_10_scores.append(1.0 if any(hit_flags[:10]) else 0.0)

            # MRR
            rr = 0.0
            for rank, flag in enumerate(hit_flags[:10], 1):
                if flag:
                    rr = 1.0 / rank
                    break
            mrr_scores.append(rr)

            # NDCG@5
            dcg = sum(h / np.log2(i + 2) for i, h in enumerate(hit_flags[:5]))
            idcg = sum(1.0 / np.log2(i + 2) for i in range(min(1, 5)))
            ndcg_5_scores.append(dcg / idcg if idcg > 0 else 0.0)

        return {
            "model": model_name,
            "model_id": model_id,
            "dim": dim,
            "precision_at_1": round(np.mean(precision_1_scores), 4),
            "precision_at_5": round(np.mean(precision_5_scores), 4),
            "recall_at_5": round(np.mean(recall_5_scores), 4),
            "recall_at_10": round(np.mean(recall_10_scores), 4),
            "mrr": round(np.mean(mrr_scores), 4),
            "ndcg_at_5": round(np.mean(ndcg_5_scores), 4),
            "throughput_chunks_sec": round(throughput, 1),
            "query_latency_ms": round(np.mean(query_latencies), 2),
            "index_size_mb": round(index_size_mb, 2),
            "n_corpus": len(corpus_df),
            "n_eval": len(eval_df),
        }

    def save_results(self, output_path: Optional[str] = None) -> str:
        if not self.results:
            logger.warning("No results to save.")
            return ""
        df = pd.DataFrame(self.results)
        path = output_path or str(BASE_DIR / "reports" / "benchmark_results.parquet")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_parquet(path, index=False)
        logger.info(f"Benchmark results saved to {path}")
        return path


if __name__ == "__main__":
    bench = EmbeddingBenchmark(n_eval_samples=100)
    # Quick test with only MiniLM
    df = bench.run(models={"MiniLM-L6-v2": "sentence-transformers/all-MiniLM-L6-v2"})
    print(df.to_string(index=False))
    bench.save_results()
