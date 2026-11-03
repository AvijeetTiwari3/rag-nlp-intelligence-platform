"""
test_hallucination_detector.py — Unit Tests for Hallucination & Faithfulness Detection
"""

import os
import sys
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.hallucination_detector import HallucinationDetector


def test_hallucination_detection_faithful():
    detector = HallucinationDetector(use_semantic=False)
    context = [{"title": "Gravity", "text": "Gravity is a fundamental interaction which causes mutual attraction between all things with mass."}]
    answer = "Gravity causes attraction between objects that have mass."

    result = detector.detect(answer, context)
    assert result["verdict"] in ["FAITHFUL", "UNCERTAIN"]
    assert result["token_faithfulness"] > 0.4
    assert result["composite_score"] > 0.4


def test_hallucination_detection_hallucinated():
    detector = HallucinationDetector(use_semantic=False)
    context = [{"title": "Biology", "text": "Photosynthesis is used by plants to convert light energy into chemical energy."}]
    answer = "Alexander the Great invaded the Persian empire in 334 BC using phalanx tactics."

    result = detector.detect(answer, context)
    assert result["verdict"] == "HALLUCINATED"
    assert result["token_faithfulness"] < 0.2
    assert len(result["flagged_claims"]) > 0
