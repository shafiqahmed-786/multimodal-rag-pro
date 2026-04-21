"""Evaluation router."""
from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

from api.schemas.schemas import EvalRequest, EvalResponse
from core.models import EvaluationSample
from evaluation.evaluator import RAGEvaluator
from config.settings import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()
router = APIRouter()

_evaluator: RAGEvaluator | None = None


def _get_evaluator() -> RAGEvaluator:
    global _evaluator
    if _evaluator is None:
        _evaluator = RAGEvaluator()
    return _evaluator


@router.post("", response_model=EvalResponse)
async def run_evaluation(payload: EvalRequest):
    """Run RAGAS-style evaluation on a batch of QA samples."""
    if not payload.samples:
        raise HTTPException(status_code=400, detail="No samples provided.")

    samples = [
        EvaluationSample(
            question=s.question,
            ground_truth=s.ground_truth,
            predicted_answer=s.predicted_answer,
            retrieved_contexts=s.retrieved_contexts,
        )
        for s in payload.samples
    ]

    try:
        result = _get_evaluator().evaluate(samples)
    except Exception as exc:
        logger.exception("Evaluation failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return EvalResponse(
        faithfulness=result.faithfulness,
        answer_relevance=result.answer_relevance,
        context_precision=result.context_precision,
        context_recall=result.context_recall,
        overall_score=result.overall_score,
        samples_evaluated=result.samples_evaluated,
        evaluated_at=result.evaluated_at,
    )
