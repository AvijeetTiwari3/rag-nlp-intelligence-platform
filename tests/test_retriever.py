"""
test_retriever.py — Unit Tests for FAISS Vector Store and Retriever
"""

import os
import sys
import pytest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.vector_store import FAISSVectorStore
from src.embedder import Embedder
from src.retriever import Retriever


def test_vector_store_add_and_search():
    dim = 64
    store = FAISSVectorStore(embedding_dim=dim)

    # Random normalized vectors
    vecs = np.random.randn(10, dim).astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)

    meta = [{"chunk_id": f"c_{i}", "title": f"Doc {i}", "text": f"Content {i}"} for i in range(10)]
    store.add(vecs, meta)

    assert store.count == 10

    # Search with the first vector
    query_vec = vecs[0:1]
    results = store.search(query_vec, top_k=3)
    assert len(results) == 3
    assert results[0]["chunk_id"] == "c_0"
    assert abs(results[0]["score"] - 1.0) < 1e-4


def test_retriever_pipeline():
    embedder = Embedder(model_name="sentence-transformers/all-MiniLM-L6-v2")
    dim = embedder.get_embedding_dim()
    store = FAISSVectorStore(embedding_dim=dim)

    passages = [
        {"chunk_id": "c_apollo", "title": "Apollo 11", "text": "Apollo 11 was the American spaceflight that first landed humans on the Moon in 1969."},
        {"chunk_id": "c_biology", "title": "Cell Biology", "text": "Mitochondria are known as the powerhouses of the cell producing ATP energy."},
    ]
    texts = [p["text"] for p in passages]
    embs = embedder.encode(texts)
    store.add(embs, passages)

    retriever = Retriever(embedder=embedder, vector_store=store)
    results = retriever.retrieve("When did astronauts land on the Moon?", top_k=1)

    assert len(results) == 1
    assert results[0]["chunk_id"] == "c_apollo"
    assert "Apollo 11" in results[0]["title"]
