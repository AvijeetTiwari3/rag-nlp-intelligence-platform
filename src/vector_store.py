"""
Production FAISS Vector Store Engine
Builds, persists, and queries dense vector representations for sub-millisecond similarity search.
"""

import os
import sys
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Tuple, Optional, Union
import faiss
import numpy as np
import pandas as pd

_pkg_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _pkg_root not in sys.path:
    sys.path.insert(0, _pkg_root)


@dataclass
class SearchResult:
    rank: int
    chunk_id: str
    source_title: str
    text: str
    similarity_score: float

    # Dictionary-like access support
    def __getitem__(self, item):
        if item in ("score", "similarity_score"):
            return self.similarity_score
        if item in ("title", "source_title"):
            return self.source_title
        if item == "text":
            return self.text
        if item == "chunk_id":
            return self.chunk_id
        if item == "rank":
            return self.rank
        raise KeyError(item)

    def get(self, item, default=None):
        try:
            return self[item]
        except KeyError:
            return default

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["title"] = self.source_title
        d["score"] = self.similarity_score
        return d


class FAISSVectorStore:
    """FAISS inner product (cosine similarity on normalized vectors) index."""

    def __init__(self, embedding_dimension: int = 384, embedding_dim: Optional[int] = None):
        if embedding_dim is not None:
            self.embedding_dimension = embedding_dim
        else:
            self.embedding_dimension = embedding_dimension
        self.index = faiss.IndexFlatIP(self.embedding_dimension)
        self.metadata_df: Optional[pd.DataFrame] = None

    @property
    def count(self) -> int:
        return self.index.ntotal

    def build_index(
        self,
        embeddings: np.ndarray,
        metadata_df: Union[pd.DataFrame, List[Dict[str, Any]]]
    ) -> None:
        """Add dense vectors and aligned metadata to index."""
        if isinstance(metadata_df, list):
            metadata_df = pd.DataFrame(metadata_df)

        if embeddings.shape[1] != self.embedding_dimension:
            raise ValueError(
                f"Embedding dim mismatch: expected {self.embedding_dimension}, got {embeddings.shape[1]}"
            )
        if len(embeddings) != len(metadata_df):
            raise ValueError("Embeddings count does not match metadata rows.")

        self.index.reset()
        self.index.add(embeddings.astype(np.float32))
        self.metadata_df = metadata_df.copy().reset_index(drop=True)

    def add(
        self,
        embeddings: np.ndarray,
        metadata: Union[pd.DataFrame, List[Dict[str, Any]]]
    ) -> None:
        """Alias for build_index / appending vectors."""
        self.build_index(embeddings, metadata)

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 5,
        threshold: float = 0.0
    ) -> List[SearchResult]:
        """Query index for top-K most similar chunks."""
        if self.index.ntotal == 0 or self.metadata_df is None:
            raise RuntimeError("Index is unpopulated. Call build_index or load first.")

        query_vec = query_vector.astype(np.float32)
        if len(query_vec.shape) == 1:
            query_vec = np.expand_dims(query_vec, axis=0)

        scores, indices = self.index.search(query_vec, top_k)
        results = []

        for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), 1):
            if idx == -1 or score < threshold:
                continue
            row = self.metadata_df.iloc[idx]
            title = row.get("source_title", row.get("title", "Unknown"))
            res = SearchResult(
                rank=rank,
                chunk_id=str(row.get("chunk_id", f"chunk_{idx}")),
                source_title=str(title),
                text=str(row.get("text", "")),
                similarity_score=round(float(score), 4)
            )
            results.append(res)

        return results

    def save(self, output_dir: str = "models", prefix: str = "faiss_index") -> None:
        """Persist FAISS index binary and metadata parquet."""
        os.makedirs(output_dir, exist_ok=True)
        index_file = os.path.join(output_dir, f"{prefix}.index")
        meta_file = os.path.join(output_dir, f"{prefix}_meta.parquet")

        faiss.write_index(self.index, index_file)
        if self.metadata_df is not None:
            self.metadata_df.to_parquet(meta_file, index=False)

    def load(
        self,
        index_path_or_dir: str = "models",
        meta_path_or_prefix: str = "faiss_index",
        dimension: int = 384
    ) -> "FAISSVectorStore":
        """Load persisted FAISS index and metadata into this store or return instance."""
        if index_path_or_dir.endswith(".index"):
            index_file = index_path_or_dir
            meta_file = meta_path_or_prefix
        else:
            index_file = os.path.join(index_path_or_dir, f"{meta_path_or_prefix}.index")
            meta_file = os.path.join(index_path_or_dir, f"{meta_path_or_prefix}_meta.parquet")

        if not os.path.exists(index_file) or not os.path.exists(meta_file):
            raise FileNotFoundError(f"FAISS index or metadata missing: {index_file}")

        self.index = faiss.read_index(index_file)
        self.metadata_df = pd.read_parquet(meta_file)
        return self

    @classmethod
    def load_index(
        cls,
        index_path_or_dir: str = "models",
        meta_path_or_prefix: str = "faiss_index",
        dimension: int = 384
    ) -> "FAISSVectorStore":
        store = cls(embedding_dimension=dimension)
        return store.load(index_path_or_dir, meta_path_or_prefix, dimension)
