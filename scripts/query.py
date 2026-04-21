#!/usr/bin/env python3
"""
CLI script — interactive query REPL against a persisted index.

Usage:
    python scripts/query.py
    python scripts/query.py --index-dir data/indexes --top-k 5
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

import logging
logging.basicConfig(level=logging.WARNING)   # quiet in REPL mode

from config.settings import get_settings
from core.rag_service import RAGService
from core.models import QueryRequest

settings = get_settings()

BANNER = """
╔══════════════════════════════════════════════════════╗
║        Multimodal RAG — Interactive Query CLI        ║
║  Type your question and press Enter. 'quit' to exit. ║
╚══════════════════════════════════════════════════════╝
"""

CONFIDENCE_COLOUR = {
    "HIGH": "\033[92m",       # green
    "MEDIUM": "\033[93m",     # yellow
    "LOW": "\033[91m",        # red
    "UNVERIFIED": "\033[90m", # grey
}
RESET = "\033[0m"


def _print_answer(answer) -> None:
    conf = answer.confidence.value.upper()
    colour = CONFIDENCE_COLOUR.get(conf, "")
    print(f"\n{colour}[{conf} confidence | {answer.latency_ms:.0f} ms]{RESET}")

    if answer.query_expansion and answer.query_expansion.rewritten != answer.query_expansion.original:
        print(f"\033[90m→ Rewritten: {answer.query_expansion.rewritten}{RESET}")

    print("\n" + "─" * 60)
    print(answer.text)
    print("─" * 60)

    if answer.verification and not answer.verification.is_faithful:
        print(f"\033[91m⚠ Faithfulness issues: {answer.verification.verification_note}{RESET}")

    if answer.citations:
        print("\nSources:")
        for i, cite in enumerate(answer.citations, 1):
            icon = {"text": "📝", "table": "📊", "image": "🖼️"}.get(cite.content_type.value, "📄")
            print(
                f"  {icon} [SOURCE_{i}] {cite.source_file} | "
                f"p.{cite.page_number} | score={cite.relevance_score:.3f}"
            )
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive RAG query CLI.")
    parser.add_argument("--index-dir", default=settings.index_dir)
    parser.add_argument("--top-k", type=int, default=settings.default_top_k)
    parser.add_argument("--no-reranking", action="store_true")
    parser.add_argument("--no-rewriting", action="store_true")
    args = parser.parse_args()

    rag = RAGService()
    try:
        rag.load_index(args.index_dir)
    except FileNotFoundError:
        print(f"❌ No index found at '{args.index_dir}'. Run scripts/ingest.py first.")
        sys.exit(1)

    print(BANNER)
    print(f"Index loaded: {rag.chunk_count} chunks")
    print(f"Model: {settings.llm_model} | Embeddings: {settings.embedding_model}\n")

    while True:
        try:
            query_text = input("❓ Question: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not query_text:
            continue
        if query_text.lower() in {"quit", "exit", "q"}:
            print("Goodbye!")
            break

        request = QueryRequest(
            text=query_text,
            top_k=args.top_k,
            enable_reranking=not args.no_reranking,
            enable_query_rewriting=not args.no_rewriting,
        )

        try:
            answer = rag.query(request)
            _print_answer(answer)
        except Exception as exc:
            print(f"\033[91mError: {exc}{RESET}\n")


if __name__ == "__main__":
    main()
