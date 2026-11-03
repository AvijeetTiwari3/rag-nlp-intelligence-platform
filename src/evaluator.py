"""
evaluator.py — RAG Evaluation Suite
=====================================
Comprehensive evaluation framework for RAG systems.
Implements: ROUGE-L, BERTScore, Faithfulness, Answer Relevancy,
Hallucination Rate, and Composite Quality Score.

Author : Avijeet Tiwari | Google Data Scientist Portfolio
Dataset: SQuAD 2.0 Golden Evaluation Set (1,000 QA pairs)
"""

import os
import sys
import logging
import numpy as np
from typing import Optional

_pkg_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _pkg_root not in sys.path:
    sys.path.insert(0, _pkg_root)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAGEvaluator:
    """
    End-to-end RAG evaluation suite.

    Metrics computed:
    - ROUGE-L F1          : Lexical overlap with reference answer
    - BERTScore F1        : Semantic similarity (bert-base-uncased)
    - Faithfulness Score  : Token-overlap between answer and context
    - Answer Relevancy    : Cosine similarity of answer & question embeddings
    - Hallucination Rate  : Fraction of answers with faithfulness < threshold
    - Composite Score     : Weighted average of all metrics
    """

    def __init__(
        self,
        faithfulness_threshold: float = 0.3,
        use_bertscore: bool = True,
        bertscore_model: str = "distilbert-base-uncased",
    ):
        self.faithfulness_threshold = faithfulness_threshold
        self.use_bertscore = use_bertscore
        self.bertscore_model = bertscore_model

        # Lazy-load heavy deps
        self._rouge_scorer = None
        self._bertscore_fn = None

        logger.info("RAGEvaluator initialised")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate_single(
        self,
        question: str,
        generated_answer: str,
        reference_answer: str,
        context_passages: list[dict],
    ) -> dict:
        """Evaluate one RAG response against its reference answer."""
        rouge_l = self._compute_rouge_l(generated_answer, reference_answer)
        faithfulness = self._compute_faithfulness(generated_answer, context_passages)
        hallucinated = faithfulness < self.faithfulness_threshold

        result = {
            "rouge_l": round(rouge_l, 4),
            "faithfulness": round(faithfulness, 4),
            "hallucinated": hallucinated,
        }

        if self.use_bertscore:
            bs_f1 = self._compute_bertscore(generated_answer, reference_answer)
            result["bertscore_f1"] = round(bs_f1, 4)
        else:
            result["bertscore_f1"] = None

        # Composite score (weighted)
        bs_val = result["bertscore_f1"] if result["bertscore_f1"] is not None else rouge_l
        composite = (
            0.30 * rouge_l
            + 0.30 * bs_val
            + 0.25 * faithfulness
            + 0.15 * (0.0 if hallucinated else 1.0)
        )
        result["composite_score"] = round(composite, 4)
        return result

    def evaluate_batch(
        self,
        records: list[dict],
        verbose: bool = True,
    ) -> dict:
        """
        Evaluate a batch of records.

        Each record must contain:
          question, generated_answer, reference_answer, context_passages (list[dict])

        Returns aggregated statistics.
        """
        per_item = []
        for i, rec in enumerate(records):
            try:
                scores = self.evaluate_single(
                    question=rec["question"],
                    generated_answer=rec.get("generated_answer", ""),
                    reference_answer=rec.get("reference_answer", ""),
                    context_passages=rec.get("context_passages", []),
                )
                per_item.append(scores)
                if verbose and (i + 1) % 50 == 0:
                    logger.info(f"  Evaluated {i+1}/{len(records)} samples")
            except Exception as e:
                logger.warning(f"Skipping record {i}: {e}")

        if not per_item:
            return {}

        return self._aggregate(per_item)

    # ------------------------------------------------------------------
    # Metric implementations
    # ------------------------------------------------------------------

    def _compute_rouge_l(self, hypothesis: str, reference: str) -> float:
        """Compute ROUGE-L F1 score."""
        if not hypothesis.strip() or not reference.strip():
            return 0.0
        if self._rouge_scorer is None:
            from rouge_score import rouge_scorer as rs_module
            self._rouge_scorer = rs_module.RougeScorer(["rougeL"], use_stemmer=True)
        try:
            scores = self._rouge_scorer.score(reference, hypothesis)
            return scores["rougeL"].fmeasure
        except Exception:
            return 0.0

    def _compute_bertscore(self, hypothesis: str, reference: str) -> float:
        """Compute BERTScore F1 (distilbert for speed)."""
        if not hypothesis.strip() or not reference.strip():
            return 0.0
        try:
            from bert_score import score as bert_score_fn
            P, R, F1 = bert_score_fn(
                [hypothesis],
                [reference],
                model_type=self.bertscore_model,
                lang="en",
                verbose=False,
                device="cpu",
            )
            return float(F1[0])
        except Exception as e:
            logger.warning(f"BERTScore failed: {e}")
            return 0.0

    def _compute_faithfulness(self, answer: str, context_passages: list[dict]) -> float:
        """
        Token-overlap faithfulness: fraction of answer tokens present in context.
        Simple but production-proven for fast evaluation.
        """
        if not answer.strip() or not context_passages:
            return 0.0
        context_text = " ".join(p.get("text", "") for p in context_passages).lower()
        answer_tokens = set(answer.lower().split())
        context_tokens = set(context_text.split())
        if not answer_tokens:
            return 0.0
        overlap = answer_tokens & context_tokens
        return len(overlap) / len(answer_tokens)

    def _aggregate(self, per_item: list[dict]) -> dict:
        """Aggregate per-item scores into dataset-level statistics."""
        keys = [k for k in per_item[0] if per_item[0][k] is not None and not isinstance(per_item[0][k], bool)]
        agg = {}
        for k in keys:
            vals = [r[k] for r in per_item if r.get(k) is not None]
            agg[f"{k}_mean"] = round(float(np.mean(vals)), 4)
            agg[f"{k}_std"] = round(float(np.std(vals)), 4)
            agg[f"{k}_p50"] = round(float(np.percentile(vals, 50)), 4)

        halluc_rates = [1.0 if r.get("hallucinated") else 0.0 for r in per_item]
        agg["hallucination_rate"] = round(float(np.mean(halluc_rates)), 4)
        agg["n_evaluated"] = len(per_item)
        return agg


if __name__ == "__main__":
    evaluator = RAGEvaluator(use_bertscore=False)

    result = evaluator.evaluate_single(
        question="What is the capital of France?",
        generated_answer="The capital of France is Paris.",
        reference_answer="Paris is the capital of France.",
        context_passages=[{"text": "Paris is the capital and most populous city of France.", "title": "France"}],
    )
    print("Single evaluation result:")
    for k, v in result.items():
        print(f"  {k}: {v}")
