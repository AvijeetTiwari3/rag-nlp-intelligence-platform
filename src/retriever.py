"""
Semantic Retriever & Information Retrieval Quality Audit Engine
Implements Precision@K, Recall@K, Mean Reciprocal Rank (MRR), and NDCG@K evaluation.
"""

import os
import sys
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Union
import numpy as np
import pandas as pd

_pkg_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _pkg_root not in sys.path:
    sys.path.insert(0, _pkg_root)

from src.embedder import EmbeddingEngine, Embedder
from src.vector_store import FAISSVectorStore, SearchResult


@dataclass
class RetrievalQualityMetrics:
    model_name: str
    total_queries_evaluated: int
    precision_at_1: float
    precision_at_3: float
    precision_at_5: float
    recall_at_5: float
    mean_reciprocal_rank: float
    ndcg_at_5: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class SemanticRetriever:
    """End-to-end dense retriever connecting Embedder and FAISS index."""

    def __init__(self, arg1: Any = None, arg2: Any = None, vector_store: Optional[FAISSVectorStore] = None, embedder: Optional[EmbeddingEngine] = None):
        # Support flexible parameter orders
        if vector_store is not None and embedder is not None:
            self.vector_store = vector_store
            self.embedder = embedder
        elif isinstance(arg1, FAISSVectorStore):
            self.vector_store = arg1
            self.embedder = arg2
        elif isinstance(arg2, FAISSVectorStore):
            self.embedder = arg1
            self.vector_store = arg2
        else:
            self.embedder = arg1
            self.vector_store = arg2

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        similarity_threshold: float = 0.0
    ) -> List[SearchResult]:
        """Embed user query and retrieve top-K nearest corpus chunks."""
        q_vec = self.embedder.encode_query(query)
        return self.vector_store.search(q_vec, top_k=top_k, threshold=similarity_threshold)


# Universal Alias
Retriever = SemanticRetriever


def evaluate_retrieval_quality(
    retriever: SemanticRetriever,
    golden_df: pd.DataFrame,
    sample_size: int = 200,
    top_k: int = 5
) -> RetrievalQualityMetrics:
    """
    Audit retrieval quality against verified ground truth answer contexts.
    Uses token overlap threshold (>=40% target token coverage) to define binary relevance.
    """
    # Check answerable column
    ans_col = "has_answer" if "has_answer" in golden_df.columns else "is_answerable"
    eval_subset = golden_df[golden_df[ans_col]].head(sample_size) if ans_col in golden_df.columns else golden_df.head(sample_size)

    p1_list, p3_list, p5_list = [], [], []
    r5_list, mrr_list, ndcg_list = [], [], []

    for _, row in eval_subset.iterrows():
        question = str(row["question"])
        gold_context_tokens = set(str(row.get("context", "")).lower().split())

        hits = retriever.retrieve(question, top_k=top_k)
        relevance_binary = []

        for h in hits:
            chunk_tokens = set(h.text.lower().split())
            overlap = len(gold_context_tokens & chunk_tokens) / max(len(chunk_tokens), 1)
            relevance_binary.append(1 if overlap >= 0.40 else 0)

        while len(relevance_binary) < top_k:
            relevance_binary.append(0)

        # Precision@K
        p1 = relevance_binary[0]
        p3 = sum(relevance_binary[:3]) / 3.0
        p5 = sum(relevance_binary[:5]) / 5.0
        r5 = 1.0 if sum(relevance_binary[:5]) > 0 else 0.0

        p1_list.append(p1)
        p3_list.append(p3)
        p5_list.append(p5)
        r5_list.append(r5)

        # MRR
        rr = 0.0
        for rank_idx, rel in enumerate(relevance_binary, 1):
            if rel == 1:
                rr = 1.0 / rank_idx
                break
        mrr_list.append(rr)

        # NDCG@5
        dcg = sum(rel / np.log2(idx + 2) for idx, rel in enumerate(relevance_binary[:5]))
        ideal_hits = sorted(relevance_binary[:5], reverse=True)
        idcg = sum(rel / np.log2(idx + 2) for idx, rel in enumerate(ideal_hits))
        ndcg = (dcg / idcg) if idcg > 0 else 0.0
        ndcg_list.append(ndcg)

    return RetrievalQualityMetrics(
        model_name=retriever.embedder.model_name,
        total_queries_evaluated=len(eval_subset),
        precision_at_1=round(float(np.mean(p1_list)), 4),
        precision_at_3=round(float(np.mean(p3_list)), 4),
        precision_at_5=round(float(np.mean(p5_list)), 4),
        recall_at_5=round(float(np.mean(r5_list)), 4),
        mean_reciprocal_rank=round(float(np.mean(mrr_list)), 4),
        ndcg_at_5=round(float(np.mean(ndcg_list)), 4)
    )
