"""
test_api.py — Integration Tests for FastAPI Endpoints
"""

import os
import sys
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from api.app import app

client = TestClient(app)


def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "RAG NLP Intelligence Platform" in data["project"]
    assert data["version"] == "1.0.0"


def test_health_endpoint():
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["version"] == "1.0.0"


def test_evaluate_endpoint():
    payload = {
        "question": "What is Python?",
        "generated_answer": "Python is a high-level programming language.",
        "reference_answer": "Python is a popular programming language.",
        "context_passages": [{"title": "Python", "text": "Python is a high-level general-purpose programming language."}]
    }
    response = client.post("/api/v1/evaluate", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "rouge_l" in data
    assert "faithfulness" in data
    assert "composite_score" in data
