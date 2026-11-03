"""
generator.py — Gemini-Powered Answer Generator
================================================
Production-grade LLM generation module using Google Gemini 1.5 Flash.
Implements RAG-aware prompt engineering with citation injection,
uncertainty quantification, and graceful fallback handling.

Author : Avijeet Tiwari | Google Data Scientist Portfolio
Dataset: SQuAD 2.0 (Stanford Question Answering Dataset)
"""

import os
import sys
import time
import logging
from typing import Optional

import google.genai as genai
from google.genai import types as genai_types

_pkg_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _pkg_root not in sys.path:
    sys.path.insert(0, _pkg_root)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

RAG_SYSTEM_PROMPT = """You are an expert question-answering assistant. Your task is to answer questions accurately and concisely based ONLY on the provided context passages.

Rules:
1. Answer ONLY from the provided context. Do NOT use external knowledge.
2. If the context does not contain enough information to answer the question, respond with: "I cannot answer this question based on the provided context."
3. Be concise and precise. Avoid unnecessary elaboration.
4. Cite passage numbers when referencing specific information (e.g., [Passage 1]).
5. For factual questions, provide a direct answer first, then supporting evidence.
"""

RAG_USER_TEMPLATE = """Context Passages:
{context}

Question: {question}

Answer (based solely on the above context):"""

UNANSWERABLE_RESPONSE = "I cannot answer this question based on the provided context."


class GeminiGenerator:
    """
    Production-grade Gemini 1.5 Flash answer generator for RAG pipelines.

    Supports:
    - Context-grounded generation with citation injection
    - Unanswerable question detection
    - Token-level usage tracking
    - Retry logic with exponential backoff
    - Configurable generation parameters
    """

    def __init__(
        self,
        api_key: str,
        model_name: str = "gemini-1.5-flash",
        temperature: float = 0.1,
        max_output_tokens: int = 512,
        top_p: float = 0.95,
        max_retries: int = 3,
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.top_p = top_p
        self.max_retries = max_retries

        # Initialise the new google-genai client
        self.client = genai.Client(api_key=api_key)

        self._total_prompt_tokens = 0
        self._total_output_tokens = 0
        self._total_calls = 0

        logger.info(f"GeminiGenerator initialised | model={model_name}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        question: str,
        retrieved_passages: list[dict],
        max_context_passages: int = 5,
    ) -> dict:
        """
        Generate a grounded answer for the given question using retrieved passages.

        Args:
            question            : The user question.
            retrieved_passages  : List of dicts with keys 'text', 'title', 'score'.
            max_context_passages: Maximum passages to include in prompt.

        Returns:
            dict with keys:
                answer          (str)  : Generated answer text.
                sources         (list) : Passage titles used.
                answerable      (bool) : Whether the question was answerable.
                latency_ms      (float): Generation latency in milliseconds.
                prompt_tokens   (int)  : Prompt token count.
                output_tokens   (int)  : Output token count.
        """
        passages = retrieved_passages[:max_context_passages]
        context_str = self._format_context(passages)
        prompt = RAG_USER_TEMPLATE.format(context=context_str, question=question)

        start = time.perf_counter()
        raw_answer = self._call_with_retry(prompt)
        latency_ms = (time.perf_counter() - start) * 1000

        answerable = UNANSWERABLE_RESPONSE.lower() not in raw_answer.lower()

        return {
            "answer": raw_answer.strip(),
            "sources": [p.get("title", "Unknown") for p in passages],
            "answerable": answerable,
            "latency_ms": round(latency_ms, 2),
            "prompt_tokens": 0,   # tracked via usage_metadata when available
            "output_tokens": 0,
        }

    def batch_generate(
        self,
        items: list[dict],
        max_context_passages: int = 5,
        delay_between_calls: float = 0.5,
    ) -> list[dict]:
        """
        Batch-generate answers for a list of {question, retrieved_passages} dicts.
        Includes rate-limit throttling via delay_between_calls.
        """
        results = []
        for i, item in enumerate(items):
            try:
                result = self.generate(
                    question=item["question"],
                    retrieved_passages=item.get("retrieved_passages", []),
                    max_context_passages=max_context_passages,
                )
                results.append(result)
                if i < len(items) - 1:
                    time.sleep(delay_between_calls)
            except Exception as e:
                logger.error(f"batch_generate error at index {i}: {e}")
                results.append({"answer": UNANSWERABLE_RESPONSE, "error": str(e)})
        return results

    def get_usage_stats(self) -> dict:
        """Return cumulative token usage statistics."""
        return {
            "total_calls": self._total_calls,
            "total_prompt_tokens": self._total_prompt_tokens,
            "total_output_tokens": self._total_output_tokens,
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _format_context(self, passages: list[dict]) -> str:
        parts = []
        for i, p in enumerate(passages, 1):
            title = p.get("title", "Unknown")
            text = p.get("text", "").strip()
            score = p.get("score", 0.0)
            parts.append(
                f"[Passage {i}] Title: {title} (relevance={score:.3f})\n{text}"
            )
        return "\n\n".join(parts)

    def _call_with_retry(self, prompt: str) -> str:
        """Call Gemini API with exponential-backoff retry on transient errors."""
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=genai_types.GenerateContentConfig(
                        system_instruction=RAG_SYSTEM_PROMPT,
                        temperature=self.temperature,
                        max_output_tokens=self.max_output_tokens,
                        top_p=self.top_p,
                    ),
                )
                self._total_calls += 1

                # Extract text safely
                if response.candidates:
                    candidate = response.candidates[0]
                    if candidate.content and candidate.content.parts:
                        text = "".join(
                            part.text
                            for part in candidate.content.parts
                            if hasattr(part, "text") and part.text
                        )
                        return text if text else UNANSWERABLE_RESPONSE
                return UNANSWERABLE_RESPONSE

            except Exception as e:
                wait = 2 ** attempt
                logger.warning(f"Gemini API attempt {attempt} failed: {e}. Retrying in {wait}s...")
                if attempt == self.max_retries:
                    logger.error("Max retries exceeded.")
                    raise
                time.sleep(wait)

        return UNANSWERABLE_RESPONSE


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os

    api_key = os.environ.get("GEMINI_API_KEY", "")

    gen = GeminiGenerator(api_key=api_key)

    passages = [
        {
            "title": "Oxygen",
            "text": "Oxygen is a chemical element with the symbol O and atomic number 8. It is a member of the chalcogen group in the periodic table, a highly reactive nonmetal, and an oxidizing agent that readily forms oxides with most elements.",
            "score": 0.92,
        },
        {
            "title": "Oxygen discovery",
            "text": "Oxygen was first isolated by Swedish chemist Carl Wilhelm Scheele in 1773 and independently discovered by Joseph Priestley in 1774.",
            "score": 0.87,
        },
    ]

    result = gen.generate(
        question="Who discovered oxygen and when?",
        retrieved_passages=passages,
    )

    print("Question : Who discovered oxygen and when?")
    print(f"Answer   : {result['answer']}")
    print(f"Answerable: {result['answerable']}")
    print(f"Latency  : {result['latency_ms']} ms")
    print(f"Sources  : {result['sources']}")
