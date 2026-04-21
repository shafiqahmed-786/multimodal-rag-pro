"""
/feedback router — thumbs up/down signal collection.

In production, these signals would be persisted to a database and used
to fine-tune retrieval weights or trigger RLHF pipelines.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from fastapi import APIRouter

from api.schemas.schemas import FeedbackPayload, FeedbackResponse
from core.models import FeedbackSignal
from config.settings import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()
router = APIRouter()

_FEEDBACK_LOG = Path(settings.data_dir) / "feedback.jsonl"


@router.post("", response_model=FeedbackResponse)
async def submit_feedback(payload: FeedbackPayload):
    """Store user feedback signal for a given answer."""
    signal = FeedbackSignal(
        answer_id=payload.answer_id,
        query_id=payload.query_id,
        rating=payload.rating,
        comment=payload.comment,
    )

    # Append to JSONL log (swap for DB write in production)
    try:
        _FEEDBACK_LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(_FEEDBACK_LOG, "a") as f:
            f.write(signal.model_dump_json() + "\n")
    except Exception as exc:
        logger.warning("Could not persist feedback: %s", exc)

    logger.info(
        "Feedback received: answer_id=%s rating=%d",
        payload.answer_id, payload.rating,
    )
    return FeedbackResponse(accepted=True, message="Feedback recorded. Thank you!")
