"""
test_embedder.py — Unit Tests for Embedder Module
"""

import os
import sys
import pytest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.embedder import Embedder


def test_embedder_initialization():
    embedder = Embedder(model_name="sentence-transformers/all-MiniLM-L6-v2")
    assert embedder.get_embedding_dim() == 384


def test_embedder_encode_single():
    embedder = Embedder(model_name="sentence-transformers/all-MiniLM-L6-v2")
    emb = embedder.encode("This is a sample document for testing.")
    assert isinstance(emb, np.ndarray)
    assert emb.shape == (1, 384)
    # Norm check (normalized embeddings should have norm ~ 1.0)
    norm = np.linalg.norm(emb[0])
    assert abs(norm - 1.0) < 1e-3


def test_embedder_encode_batch():
    embedder = Embedder(model_name="sentence-transformers/all-MiniLM-L6-v2")
    texts = [
        "First Wikipedia chunk about science.",
        "Second chunk about world history.",
        "Third chunk about deep learning.",
    ]
    embs = embedder.encode(texts, batch_size=2)
    assert isinstance(embs, np.ndarray)
    assert embs.shape == (3, 384)
