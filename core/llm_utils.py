"""
LLM utility helpers shared across all pipeline modules.

Provides:
- llm_call_with_retry: exponential backoff with fallback
- extract_json: robust JSON extraction from LLM output that may contain
  preamble, markdown fences, or trailing text
"""
from __future__ import annotations

import json
import logging
import random
import re
import time
from typing import Callable, TypeVar

logger = logging.getLogger(__name__)
T = TypeVar("T")


# ---------------------------------------------------------------------------
# Robust JSON extraction
# ---------------------------------------------------------------------------

def extract_json(text: str) -> dict:
    """
    Robustly extract the first JSON object from LLM output.
    Handles: markdown fences, "Sure! Here's the JSON:" preamble, trailing text.
    """
    # Strip markdown fences
    text = re.sub(r"```(?:json)?\s*", "", text).strip().rstrip("`").strip()

    # Direct parse attempt
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Find first { and match to closing }
    start = text.find("{")
    if start == -1:
        raise ValueError(f"No JSON object found in LLM output: {text[:200]!r}")

    depth = 0
    for i, ch in enumerate(text[start:], start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start: i + 1])
                except json.JSONDecodeError:
                    break

    raise ValueError(f"Could not parse JSON from LLM output: {text[:200]!r}")


# ---------------------------------------------------------------------------
# Retry wrapper
# ---------------------------------------------------------------------------

def llm_call_with_retry(
    call_fn: Callable[[], str],
    max_retries: int = 3,
    base_delay: float = 1.0,
    fallback_fn: Callable[[], str] | None = None,
) -> str:
    """
    Execute an LLM call with exponential backoff on transient errors.

    Retries on: 429 (rate limit), 503 (service unavailable), network errors.
    Does NOT retry on: 400, 401, 403, 404 (non-transient client errors).

    Falls back to fallback_fn if all retries are exhausted.
    Raises RuntimeError if no fallback is available.
    """
    last_exc: Exception | None = None

    for attempt in range(max_retries):
        try:
            return call_fn()
        except Exception as exc:
            last_exc = exc

            # Determine HTTP status if available
            status = _extract_status(exc)

            # Non-retryable 4xx (except 429 rate limit)
            if status and 400 <= status < 500 and status != 429:
                logger.warning("Non-retryable LLM error %s: %s", status, exc)
                break

            delay = base_delay * (2 ** attempt) + random.uniform(0, 0.5)
            logger.warning(
                "LLM call failed (attempt %d/%d), retrying in %.1fs: %s",
                attempt + 1, max_retries, delay, exc,
            )
            time.sleep(delay)

    # All retries exhausted
    if fallback_fn is not None:
        logger.warning("All LLM retries failed, using fallback. Last error: %s", last_exc)
        return fallback_fn()

    raise RuntimeError(
        f"LLM call failed after {max_retries} retries: {last_exc}"
    ) from last_exc


def _extract_status(exc: Exception) -> int | None:
    """Extract HTTP status code from a google-genai or openai exception."""
    # google-genai: exception has .status_code or .code
    for attr in ("status_code", "code"):
        val = getattr(exc, attr, None)
        if isinstance(val, int):
            return val
    # Try nested response object
    resp = getattr(exc, "response", None)
    if resp is not None:
        for attr in ("status_code", "status"):
            val = getattr(resp, attr, None)
            if isinstance(val, int):
                return val
    return None
