"""
hallucination_detector.py — NLI-Based Faithfulness Auditor
============================================================
Detects hallucinations in RAG-generated answers by measuring
semantic entailment between the generated answer and source context.

Uses token-overlap + semantic similarity for production-grade
hallucination detection without requiring heavy NLI models.

Author : Avijeet Tiwari | Google Data Scientist Portfolio
"""

import os
import sys
import re
import logging
from typing import Optional

_pkg_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _pkg_root not in sys.path:
    sys.path.insert(0, _pkg_root)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HallucinationDetector:
    """
    Production-grade hallucination detector for RAG systems.

    Detection pipeline:
    1. Token-overlap faithfulness (fast, no model)
    2. Named entity verification (checks if entities in answer appear in context)
    3. Numeric claim verification (numbers in answer vs context)
    4. Semantic entailment score (optional, uses SentenceTransformer cosine sim)

    Verdict: HALLUCINATED | FAITHFUL | UNCERTAIN
    """

    VERDICTS = {"HALLUCINATED": 0, "UNCERTAIN": 1, "FAITHFUL": 2}

    def __init__(
        self,
        token_threshold: float = 0.4,
        entity_threshold: float = 0.5,
        numeric_threshold: float = 0.8,
        use_semantic: bool = True,
        semantic_threshold: float = 0.5,
        embedder=None,
    ):
        self.token_threshold = token_threshold
        self.entity_threshold = entity_threshold
        self.numeric_threshold = numeric_threshold
        self.use_semantic = use_semantic
        self.semantic_threshold = semantic_threshold
        self._embedder = embedder  # optional pre-built Embedder instance

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, answer: str, context_passages: list[dict]) -> dict:
        """
        Detect hallucination in a single answer given context passages.

        Returns:
            dict with keys:
                verdict           (str)   : FAITHFUL | UNCERTAIN | HALLUCINATED
                token_faithfulness (float): Token-overlap score
                entity_score      (float): NE coverage score
                numeric_score     (float): Numeric claim coverage
                semantic_score    (float): Cosine similarity (if enabled)
                composite_score   (float): Weighted ensemble
                flagged_claims    (list)  : Specific phrases that appear hallucinated
        """
        context_text = " ".join(p.get("text", "") for p in context_passages)

        token_f = self._token_faithfulness(answer, context_text)
        entity_s = self._entity_coverage(answer, context_text)
        numeric_s = self._numeric_coverage(answer, context_text)
        flagged = self._find_flagged_claims(answer, context_text)

        semantic_s = None
        if self.use_semantic and self._embedder is not None:
            semantic_s = self._semantic_similarity(answer, context_text)

        # Composite score
        if semantic_s is not None:
            composite = 0.30 * token_f + 0.25 * entity_s + 0.20 * numeric_s + 0.25 * semantic_s
        else:
            composite = 0.40 * token_f + 0.35 * entity_s + 0.25 * numeric_s

        # Verdict
        if composite >= 0.65:
            verdict = "FAITHFUL"
        elif composite >= 0.40:
            verdict = "UNCERTAIN"
        else:
            verdict = "HALLUCINATED"

        return {
            "verdict": verdict,
            "token_faithfulness": round(token_f, 4),
            "entity_score": round(entity_s, 4),
            "numeric_score": round(numeric_s, 4),
            "semantic_score": round(semantic_s, 4) if semantic_s is not None else None,
            "composite_score": round(composite, 4),
            "flagged_claims": flagged,
        }

    def detect_batch(self, records: list[dict]) -> list[dict]:
        """
        Batch detection. Each record: {answer, context_passages}.
        Returns list of detection results.
        """
        results = []
        for rec in records:
            try:
                r = self.detect(rec.get("answer", ""), rec.get("context_passages", []))
                results.append(r)
            except Exception as e:
                logger.warning(f"Detection failed: {e}")
                results.append({"verdict": "UNCERTAIN", "error": str(e)})
        return results

    def hallucination_rate(self, records: list[dict]) -> float:
        """Return fraction of records flagged as HALLUCINATED."""
        results = self.detect_batch(records)
        halluc = sum(1 for r in results if r.get("verdict") == "HALLUCINATED")
        return halluc / len(results) if results else 0.0

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _token_faithfulness(self, answer: str, context: str) -> float:
        a_tokens = set(self._tokenise(answer))
        c_tokens = set(self._tokenise(context))
        if not a_tokens:
            return 1.0
        return len(a_tokens & c_tokens) / len(a_tokens)

    def _entity_coverage(self, answer: str, context: str) -> float:
        """Check if capitalised (named-entity-like) tokens in answer appear in context."""
        ne_pattern = re.compile(r"\b[A-Z][a-zA-Z]{2,}\b")
        answer_nes = set(ne_pattern.findall(answer))
        if not answer_nes:
            return 1.0  # no named entities to verify
        context_upper = set(ne_pattern.findall(context))
        covered = answer_nes & context_upper
        return len(covered) / len(answer_nes)

    def _numeric_coverage(self, answer: str, context: str) -> float:
        """Check if numeric claims in answer appear in context."""
        num_pattern = re.compile(r"\b\d+(?:\.\d+)?\b")
        answer_nums = set(num_pattern.findall(answer))
        if not answer_nums:
            return 1.0
        context_nums = set(num_pattern.findall(context))
        covered = answer_nums & context_nums
        return len(covered) / len(answer_nums)

    def _semantic_similarity(self, answer: str, context: str) -> float:
        """Cosine similarity using pre-built embedder."""
        try:
            emb_a = self._embedder.encode([answer])[0]
            # Use first 512 chars of context for speed
            emb_c = self._embedder.encode([context[:512]])[0]
            dot = float(emb_a @ emb_c)
            norm = float(
                (emb_a @ emb_a) ** 0.5 * (emb_c @ emb_c) ** 0.5
            ) + 1e-9
            return max(0.0, min(1.0, dot / norm))
        except Exception:
            return 0.5  # neutral fallback

    def _find_flagged_claims(self, answer: str, context: str, n: int = 5) -> list[str]:
        """Return short phrases in answer that have no token-overlap with context."""
        context_tokens = set(self._tokenise(context))
        sentences = re.split(r"[.!?;]", answer)
        flagged = []
        for sent in sentences:
            s = sent.strip()
            if not s:
                continue
            tokens = set(self._tokenise(s))
            if tokens and len(tokens & context_tokens) / len(tokens) < 0.3:
                flagged.append(s[:120])
            if len(flagged) >= n:
                break
        return flagged

    @staticmethod
    def _tokenise(text: str) -> list[str]:
        return re.findall(r"\b\w+\b", text.lower())


if __name__ == "__main__":
    detector = HallucinationDetector(use_semantic=False)

    context = [{"text": "The Eiffel Tower is located in Paris, France. It was built in 1889 by Gustave Eiffel.", "title": "Eiffel Tower"}]

    # Faithful answer
    r1 = detector.detect("The Eiffel Tower was built in 1889 by Gustave Eiffel in Paris.", context)
    print(f"Faithful test  -> verdict={r1['verdict']}, composite={r1['composite_score']}")

    # Hallucinated answer
    r2 = detector.detect("The Eiffel Tower was built in 1750 by Napoleon Bonaparte in Lyon.", context)
    print(f"Hallucinated test -> verdict={r2['verdict']}, composite={r2['composite_score']}")
    print(f"Flagged claims: {r2['flagged_claims']}")
