"""
test_evaluator.py — Unit Tests for RAG Evaluator
"""

import os
import sys
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.evaluator import RAGEvaluator


def test_evaluator_rouge_l():
    evaluator = RAGEvaluator(use_bertscore=False)
    hypothesis = "The capital of France is Paris."
    reference = "Paris is the capital of France."
    context = [{"title": "France", "text": "Paris is the capital of France and its largest city."}]

    metrics = evaluator.evaluate_single(
        question="What is France's capital?",
        generated_answer=hypothesis,
        reference_answer=reference,
        context_passages=context,
    )
    assert metrics["rouge_l"] > 0.6
    assert metrics["faithfulness"] > 0.5
    assert not metrics["hallucinated"]
    assert metrics["composite_score"] > 0.5
