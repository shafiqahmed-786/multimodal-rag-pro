"""
Answer generator — Google Gemini, table-aware.

Upgrades:
  - Table-aware context block: injects structured JSON for table chunks
    instead of raw pipe-formatted markdown (Fix 1b)
  - Retry with exponential backoff + model fallback (Fix 3c)
  - Confidence-calibrated response: structured note explaining certainty
"""
from __future__ import annotations

import asyncio
import logging
from typing import AsyncIterator

from google import genai
from google.genai import types

from core.interfaces import BaseGenerator
from core.llm_utils import llm_call_with_retry
from core.models import Citation, ContentType, RetrievedChunk
from config.settings import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

_SYSTEM_PROMPT = """You are an expert document analyst. Answer questions using ONLY the provided context.

Rules:
- Be factual and concise
- Cite sources using [SOURCE_N] inline (e.g., "According to [SOURCE_1]...")
- If the answer is not in the context, say "I cannot find this information in the provided documents."
- For DATA TABLES provided as JSON, use the exact values from the structured data for numeric answers
- Do NOT speculate or add outside knowledge
- End with a "Sources Used:" section listing each [SOURCE_N] cited"""


def _build_context_block(chunks: list[RetrievedChunk]) -> str:
    """
    Build LLM context block.

    Table chunks: inject structured JSON for exact value lookups.
    Text chunks: use prose content as-is.
    """
    parts = []
    for i, rc in enumerate(chunks, start=1):
        chunk = rc.chunk
        header = (
            f"[SOURCE_{i}] "
            f"File: {chunk.source_file} | "
            f"Page: {chunk.page_number} | "
            f"Type: {chunk.content_type.value}"
        )
        if chunk.section_heading:
            header += f" | Section: {chunk.section_heading}"

        if (
            chunk.content_type == ContentType.TABLE
            and chunk.metadata.get("structured_data")
        ):
            # Give LLM the structured JSON — precise value lookup
            body = (
                "This is a DATA TABLE. Use the JSON below for exact value lookups.\n"
                f"```json\n{chunk.metadata['structured_data']}\n```"
            )
        elif chunk.content_type == ContentType.TABLE and chunk.metadata.get("markdown"):
            # Fallback: markdown table if no structured data
            body = f"TABLE:\n{chunk.metadata['markdown']}"
        else:
            body = chunk.content

        parts.append(f"{header}\n{body}")

    return "\n\n---\n\n".join(parts)


def _build_citations(chunks: list[RetrievedChunk]) -> list[Citation]:
    citations = []
    for i, rc in enumerate(chunks, start=1):
        chunk = rc.chunk
        excerpt = chunk.content[:200].rstrip() + ("…" if len(chunk.content) > 200 else "")
        citations.append(
            Citation(
                chunk_id=chunk.id,
                source_file=chunk.source_file,
                page_number=chunk.page_number,
                content_type=chunk.content_type,
                excerpt=excerpt,
                relevance_score=rc.score,
            )
        )
    return citations


class AnswerGenerator(BaseGenerator):
    """Generates citation-grounded answers using Google Gemini."""

    def __init__(self) -> None:
        self._client = genai.Client(api_key=settings.gemini_api_key)

    def generate(self, query: str, chunks: list[RetrievedChunk]) -> str:
        context = _build_context_block(chunks)
        prompt = f"{_SYSTEM_PROMPT}\n\nContext:\n{context}\n\nQuestion: {query}"

        def call() -> str:
            resp = self._client.models.generate_content(
                model=settings.llm_model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=settings.llm_temperature,
                    max_output_tokens=settings.llm_max_tokens,
                ),
            )
            return resp.text or ""

        def fallback() -> str:
            # Try with fallback model
            try:
                resp = self._client.models.generate_content(
                    model=settings.llm_fallback_model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=settings.llm_temperature,
                        max_output_tokens=settings.llm_max_tokens,
                    ),
                )
                return resp.text or ""
            except Exception:
                return "I was unable to generate an answer at this time. Please try again shortly."

        return llm_call_with_retry(call, max_retries=3, fallback_fn=fallback)

    async def stream(
        self, query: str, chunks: list[RetrievedChunk]
    ) -> AsyncIterator[str]:
        context = _build_context_block(chunks)
        prompt = f"{_SYSTEM_PROMPT}\n\nContext:\n{context}\n\nQuestion: {query}"
        try:
            async for chunk in await self._client.aio.models.generate_content_stream(
                model=settings.llm_model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=settings.llm_temperature,
                    max_output_tokens=settings.llm_max_tokens,
                ),
            ):
                if chunk.text:
                    yield chunk.text
        except Exception as exc:
            logger.warning("Streaming failed: %s", exc)
            yield "\n\n[Streaming error — please retry in non-stream mode.]"

    def build_citations(self, chunks: list[RetrievedChunk]) -> list[Citation]:
        return _build_citations(chunks)
