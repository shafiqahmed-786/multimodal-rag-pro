"""
Evaluation pipeline — RAGAS-style metrics computed via LLM judges (migrated to Google Gemini).

Metrics:
  - faithfulness      : is the answer grounded in the retrieved context?
  - answer_relevance  : does the answer address the question?
  - context_precision : are the retrieved chunks actually useful?
  - context_recall    : does the context contain the answer to the question?
"""
from __future__ import annotations

import json
import logging
from statistics import mean
from typing import Optional

from google import genai
from google.genai import types

from core.models import EvaluationResult, EvaluationSample
from config.settings import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

_JUDGE_SYSTEM = """You are an expert evaluator for RAG systems.
Score the provided metric from 0.0 to 1.0 (float).
Respond ONLY with valid JSON: {"score": <float>, "reason": "<brief reason>"}"""


def _llm_score(prompt: str, client: genai.Client) -> float:
    try:
        resp = client.models.generate_content(
            model=settings.llm_model,
            contents=f"{_JUDGE_SYSTEM}\n\n{prompt}",
            config=types.GenerateContentConfig(
                temperature=0.0,
                max_output_tokens=128,
            ),
        )
        raw = resp.text or "{}"
        raw = raw.strip().removeprefix("```json").removesuffix("```").strip()
        return float(json.loads(raw).get("score", 0.5))
    except Exception as exc:  # noqa: BLE001
        logger.warning("LLM judge failed: %s", exc)
        return 0.5


class RAGEvaluator:
    """Automated evaluation pipeline for a RAG system."""

    def __init__(self) -> None:
        self._client = genai.Client(api_key=settings.gemini_api_key)

    # ---------------------------------------------------------------- metrics

    def faithfulness(self, sample: EvaluationSample) -> float:
        context = "\n\n".join(sample.retrieved_contexts[:5])
        prompt = (
            f"QUESTION: {sample.question}\n\n"
            f"ANSWER: {sample.predicted_answer}\n\n"
            f"CONTEXT: {context}\n\n"
            "Score 0-1: How much of the answer is supported by the context?"
        )
        return _llm_score(prompt, self._client)

    def answer_relevance(self, sample: EvaluationSample) -> float:
        prompt = (
            f"QUESTION: {sample.question}\n\n"
            f"ANSWER: {sample.predicted_answer}\n\n"
            "Score 0-1: How well does the answer address the question?"
        )
        return _llm_score(prompt, self._client)

    def context_precision(self, sample: EvaluationSample) -> float:
        context = "\n\n".join(sample.retrieved_contexts[:5])
        prompt = (
            f"QUESTION: {sample.question}\n\n"
            f"CONTEXT: {context}\n\n"
            "Score 0-1: What fraction of the retrieved context is relevant to the question?"
        )
        return _llm_score(prompt, self._client)

    def context_recall(self, sample: EvaluationSample) -> float:
        context = "\n\n".join(sample.retrieved_contexts[:5])
        prompt = (
            f"QUESTION: {sample.question}\n\n"
            f"GROUND TRUTH ANSWER: {sample.ground_truth}\n\n"
            f"CONTEXT: {context}\n\n"
            "Score 0-1: Does the context contain all information needed for the ground truth answer?"
        )
        return _llm_score(prompt, self._client)

    # ---------------------------------------------------------------- pipeline

    def evaluate(self, samples: list[EvaluationSample]) -> EvaluationResult:
        faithfulness_scores: list[float] = []
        relevance_scores: list[float] = []
        precision_scores: list[float] = []
        recall_scores: list[float] = []

        for i, sample in enumerate(samples):
            logger.info("Evaluating sample %d/%d…", i + 1, len(samples))
            faithfulness_scores.append(self.faithfulness(sample))
            relevance_scores.append(self.answer_relevance(sample))
            precision_scores.append(self.context_precision(sample))
            recall_scores.append(self.context_recall(sample))

        f = mean(faithfulness_scores) if faithfulness_scores else 0.0
        r = mean(relevance_scores) if relevance_scores else 0.0
        p = mean(precision_scores) if precision_scores else 0.0
        rc = mean(recall_scores) if recall_scores else 0.0

        return EvaluationResult(
            faithfulness=round(f, 4),
            answer_relevance=round(r, 4),
            context_precision=round(p, 4),
            context_recall=round(rc, 4),
            overall_score=round(mean([f, r, p, rc]), 4),
            samples_evaluated=len(samples),
        )
