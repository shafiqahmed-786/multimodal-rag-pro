"""
Hallucination verifier — LLM self-check (Google Gemini).

Upgrades:
  - Uses llm_fallback_model (cheaper/faster) — reduces 503 frequency
  - Retry with exponential backoff
  - Robust JSON extraction
  - Tighter token limits (300 chars × 3 chunks vs 400 × 5 originally)
  - Graceful safety-filter handling (400 with safety reason)
  - Human-readable fallback note instead of leaking raw exception
"""
from __future__ import annotations

import json
import logging

from google import genai
from google.genai import types

from core.interfaces import BaseVerifier
from core.llm_utils import extract_json, llm_call_with_retry
from core.models import AnswerConfidence, RetrievedChunk, VerificationResult
from config.settings import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

_VERIFY_SYSTEM = """You are a hallucination detector for a RAG system.

Given a QUESTION, an ANSWER, and SOURCE CHUNKS, determine:
1. Is every factual claim in the answer supported by the sources?
2. Are there any statements that go beyond or contradict the sources?

Respond ONLY with valid JSON:
{
  "is_faithful": true or false,
  "confidence": "high" | "medium" | "low",
  "issues": ["issue 1"],
  "verification_note": "brief summary"
}"""

_CONFIDENCE_MAP = {
    "high": AnswerConfidence.HIGH,
    "medium": AnswerConfidence.MEDIUM,
    "low": AnswerConfidence.LOW,
}


class AnswerVerifier(BaseVerifier):
    """LLM-based faithfulness checker using the fallback model for speed."""

    def __init__(self) -> None:
        self._client = genai.Client(api_key=settings.gemini_api_key)

    def verify(
        self, query: str, answer: str, chunks: list[RetrievedChunk]
    ) -> VerificationResult:
        if not settings.enable_verification:
            return VerificationResult(
                is_faithful=True,
                confidence=AnswerConfidence.UNVERIFIED,
                verification_note="Verification disabled.",
            )

        # Limit context to avoid token overflows (300 chars × max 3 chunks)
        context = "\n\n".join(
            f"[S{i + 1}]: {rc.chunk.content[:300]}"
            for i, rc in enumerate(chunks[:3])
        )
        user_message = (
            f"QUESTION: {query}\n\n"
            f"ANSWER: {answer[:500]}\n\n"
            f"SOURCE CHUNKS:\n{context}"
        )

        def call() -> str:
            resp = self._client.models.generate_content(
                # Use fallback model — cheaper, faster, less likely to 503
                model=settings.llm_fallback_model,
                contents=f"{_VERIFY_SYSTEM}\n\n{user_message}",
                config=types.GenerateContentConfig(
                    temperature=0.0,
                    max_output_tokens=150,
                ),
            )
            return resp.text or "{}"

        def fallback() -> str:
            return json.dumps({
                "is_faithful": True,
                "confidence": "low",
                "issues": [],
                "verification_note": "Verification service unavailable — not checked.",
            })

        try:
            raw = llm_call_with_retry(call, max_retries=2, fallback_fn=fallback)
            parsed = extract_json(raw)

            return VerificationResult(
                is_faithful=bool(parsed.get("is_faithful", True)),
                confidence=_CONFIDENCE_MAP.get(
                    parsed.get("confidence", "low"), AnswerConfidence.LOW
                ),
                issues=parsed.get("issues", []),
                verification_note=parsed.get(
                    "verification_note", "Verification completed."
                ),
            )

        except Exception as exc:
            logger.warning("Verifier failed entirely: %s", exc)
            return VerificationResult(
                is_faithful=True,
                confidence=AnswerConfidence.UNVERIFIED,
                verification_note="Verification service unavailable.",
            )
