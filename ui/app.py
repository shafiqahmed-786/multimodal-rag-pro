"""
Streamlit UI — production-grade interface.

Features:
  - File upload with progress tracking
  - Answer streaming (token-by-token)
  - Source citation cards with modality icons
  - Retrieval diagnostics expandable panel
  - Evaluation dashboard tab
  - Thumbs up / thumbs down feedback
  - Confidence badge
  - Index persistence (auto-save after ingest)
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import streamlit as st

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import get_settings
from core.models import QueryRequest, RetrievalMode
from core.rag_service import RAGService
from evaluation.evaluator import RAGEvaluator
from core.models import EvaluationSample

settings = get_settings()

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Multimodal RAG",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------
st.markdown(
    """
<style>
.confidence-HIGH   { color: #22c55e; font-weight: 700; }
.confidence-MEDIUM { color: #f59e0b; font-weight: 700; }
.confidence-LOW    { color: #ef4444; font-weight: 700; }
.confidence-UNVERIFIED { color: #94a3b8; font-weight: 700; }
.citation-card {
    border-left: 4px solid #6366f1;
    padding: 0.5rem 1rem;
    background: #f8fafc;
    border-radius: 4px;
    margin-bottom: 0.5rem;
    font-size: 0.85rem;
}
.chunk-badge {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 12px;
    font-size: 0.75rem;
    font-weight: 600;
    margin-right: 4px;
}
.badge-text  { background: #dbeafe; color: #1d4ed8; }
.badge-table { background: #dcfce7; color: #15803d; }
.badge-image { background: #fce7f3; color: #be185d; }
</style>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "rag" not in st.session_state:
    st.session_state.rag = RAGService()
    # Auto-load persisted index
    index_path = Path(settings.index_dir)
    if (index_path / "dense.index").exists():
        try:
            st.session_state.rag.load_index()
        except Exception:
            pass

if "last_answer" not in st.session_state:
    st.session_state.last_answer = None

rag: RAGService = st.session_state.rag

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/document.png", width=60)
    st.title("Multimodal RAG")
    st.caption("v2.0 — Production Grade")

    st.divider()
    st.subheader("📄 Ingest Document")
    uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

    if uploaded_file and st.button("⚡ Build Index", use_container_width=True):
        import tempfile, shutil
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            shutil.copyfileobj(uploaded_file, tmp)
            tmp_path = tmp.name

        progress_bar = st.progress(0.0, text="Starting…")

        def _cb(msg: str, frac: float):
            progress_bar.progress(min(frac, 1.0), text=msg)

        with st.spinner("Ingesting…"):
            result = rag.ingest(tmp_path, progress_callback=_cb)
            rag.save_index()
            Path(tmp_path).unlink(missing_ok=True)

        progress_bar.empty()
        st.success(f"✅ {result['chunks_indexed']} chunks indexed from **{uploaded_file.name}**")

    st.divider()
    st.subheader("⚙️ Retrieval Settings")
    top_k = st.slider("Results (top-k)", 1, 15, settings.default_top_k)
    enable_reranking = st.toggle("Cross-encoder reranking", value=settings.enable_reranking)
    enable_rewriting = st.toggle("Query rewriting", value=settings.enable_query_rewriting)
    show_debug = st.toggle("Show retrieval diagnostics", value=False)

    st.divider()
    if rag.is_ready:
        st.metric("Chunks indexed", rag.chunk_count)
        st.success("Index ready")
    else:
        st.warning("No index — upload a PDF first.")

# ---------------------------------------------------------------------------
# Main tabs
# ---------------------------------------------------------------------------
tab_qa, tab_eval = st.tabs(["💬 Q&A", "📊 Evaluation"])

# ===========================================================================
# Q&A Tab
# ===========================================================================
with tab_qa:
    st.header("Ask a Question")
    query_text = st.text_area(
        "Your question",
        placeholder="What is the total GDP of Qatar in 2022?",
        height=80,
    )

    col_ask, col_stream = st.columns([2, 1])
    with col_ask:
        ask_btn = st.button("🔍 Ask", use_container_width=True, type="primary")
    with col_stream:
        stream_mode = st.checkbox("Stream answer", value=False)

    if ask_btn and query_text:
        if not rag.is_ready:
            st.error("Please upload and index a document first.")
        else:
            request = QueryRequest(
                text=query_text,
                top_k=top_k,
                retrieval_mode=RetrievalMode.HYBRID,
                enable_reranking=enable_reranking,
                enable_query_rewriting=enable_rewriting,
                stream=stream_mode,
            )

            # -------- streaming path --------
            if stream_mode:
                import asyncio

                st.subheader("Answer")
                answer_placeholder = st.empty()
                full_text_list = [""]

                async def _run_stream():
                    async for token in rag.query_stream(request):
                        full_text_list[0] += token
                        answer_placeholder.markdown(full_text_list[0] + "▌")
                    answer_placeholder.markdown(full_text_list[0])

                asyncio.run(_run_stream())
                st.session_state.last_answer = None   # no structured answer in stream mode

            # -------- sync path --------
            else:
                with st.spinner("Thinking…"):
                    answer = rag.query(request)
                    st.session_state.last_answer = answer

    # -------- Display answer --------
    answer = st.session_state.last_answer
    if answer:
        # Confidence badge
        conf = answer.confidence.value.upper()
        conf_cls = f"confidence-{conf}"
        st.markdown(
            f"**Confidence:** <span class='{conf_cls}'>{conf}</span> &nbsp;|&nbsp; "
            f"**Latency:** {answer.latency_ms:.0f} ms",
            unsafe_allow_html=True,
        )

        # Query expansion info
        if answer.query_expansion and answer.query_expansion.rewritten != answer.query_expansion.original:
            with st.expander("🔄 Query rewriting"):
                st.write(f"**Original:** {answer.query_expansion.original}")
                st.write(f"**Rewritten:** {answer.query_expansion.rewritten}")
                if answer.query_expansion.sub_queries:
                    st.write("**Sub-queries:**")
                    for sq in answer.query_expansion.sub_queries:
                        st.write(f"  • {sq}")

        st.divider()
        st.subheader("Answer")
        st.markdown(answer.text)

        # Verification
        if answer.verification:
            v = answer.verification
            if not v.is_faithful:
                st.warning(
                    f"⚠️ Potential hallucination detected: {v.verification_note}\n\n"
                    + "\n".join(f"• {i}" for i in v.issues)
                )
            else:
                st.success(f"✅ Answer verified faithful — {v.verification_note}")

        # Feedback
        st.divider()
        col_up, col_dn, _ = st.columns([1, 1, 8])
        with col_up:
            if st.button("👍", key="thumb_up"):
                st.toast("Thanks for the positive feedback!")
        with col_dn:
            if st.button("👎", key="thumb_dn"):
                st.toast("Thanks — we'll use this to improve.")

        # Citations
        st.divider()
        st.subheader("📚 Sources")
        for i, cite in enumerate(answer.citations, start=1):
            modality_cls = f"badge-{cite.content_type.value}"
            modality_icon = {"text": "📝", "table": "📊", "image": "🖼️"}.get(
                cite.content_type.value, "📄"
            )
            st.markdown(
                f"""<div class="citation-card">
                <span class="chunk-badge {modality_cls}">{modality_icon} {cite.content_type.value}</span>
                <strong>[SOURCE_{i}]</strong> {cite.source_file} — Page {cite.page_number}
                &nbsp;|&nbsp; relevance: {cite.relevance_score:.3f}<br/>
                <em>{cite.excerpt}</em>
                </div>""",
                unsafe_allow_html=True,
            )

        # Retrieval diagnostics
        if show_debug:
            st.divider()
            with st.expander("🔬 Retrieval Diagnostics", expanded=True):
                debug = answer.retrieval_debug
                plan = debug.get("plan", {})

                # Pipeline plan summary
                if plan:
                    pcols = st.columns(4)
                    pcols[0].metric("Query plan", "complex" if plan.get("rewrite") else "factual")
                    pcols[1].metric("Rewrite", "✅" if plan.get("rewrite") else "⏭ skipped")
                    pcols[2].metric("Rerank", "✅" if plan.get("rerank") else "⏭ skipped")
                    pcols[3].metric("Direct answer", "✅ yes" if debug.get("direct_answer_used") else "no")

                st.divider()
                cols = st.columns(3)
                cols[0].metric("Chunks retrieved", debug.get("num_retrieved", 0))
                cols[1].metric("After reranking", debug.get("num_after_rerank", 0))
                cols[2].metric("Final passed to LLM", len(answer.retrieved_chunks))

                st.write("**Retrieved chunks (ranked):**")
                for rc in answer.retrieved_chunks:
                    icon = {"text": "📝", "table": "📊", "image": "🖼️"}.get(
                        rc.chunk.content_type.value, "📄"
                    )
                    expl = rc.retrieval_explanation
                    boost_tag = f" | 🚀 {expl.get('boost_applied')}" if expl.get("boost_applied") else ""
                    with st.expander(
                        f"Rank {rc.rank+1} | {icon} {rc.chunk.content_type.value} | "
                        f"p.{rc.chunk.page_number} | score={rc.score:.4f}{boost_tag}"
                    ):
                        if rc.chunk.section_heading:
                            st.caption(f"Section: {rc.chunk.section_heading}")
                        st.text(rc.chunk.content[:600])
                        if expl:
                            st.caption(
                                f"Dense: {expl.get('dense_score', 'n/a')} | "
                                f"Sparse: {expl.get('sparse_score', 'n/a')} | "
                                f"RRF: {expl.get('rrf_score', 'n/a')} | "
                                f"Method: {rc.retrieval_method} | "
                                f"Source: {rc.chunk.source_file}"
                            )
                        else:
                            st.caption(f"Method: {rc.retrieval_method} | Source: {rc.chunk.source_file}")

# ===========================================================================
# Evaluation Tab
# ===========================================================================
with tab_eval:
    st.header("📊 Evaluation Dashboard")
    st.info(
        "Paste QA pairs below to evaluate the RAG system on RAGAS-style metrics. "
        "Each row: question | ground_truth | predicted_answer"
    )

    eval_text = st.text_area(
        "QA samples (pipe-separated: question | ground_truth | predicted_answer)",
        height=200,
        placeholder="What is Qatar's GDP? | Qatar GDP was $179B in 2021 | Qatar's GDP was approximately $179 billion in 2021.",
    )

    if st.button("▶️ Run Evaluation", use_container_width=True):
        lines = [l.strip() for l in eval_text.strip().split("\n") if l.strip()]
        samples = []
        errors = []
        for line in lines:
            parts = [p.strip() for p in line.split("|")]
            if len(parts) < 3:
                errors.append(f"Skipped (not enough columns): {line[:60]}")
                continue
            # Use retrieved chunks from the current answer as context (if available)
            contexts = []
            if st.session_state.last_answer:
                contexts = [rc.chunk.content for rc in st.session_state.last_answer.retrieved_chunks]
            samples.append(
                EvaluationSample(
                    question=parts[0],
                    ground_truth=parts[1],
                    predicted_answer=parts[2],
                    retrieved_contexts=contexts,
                )
            )

        if errors:
            for e in errors:
                st.warning(e)

        if not samples:
            st.error("No valid samples found.")
        else:
            with st.spinner(f"Evaluating {len(samples)} samples…"):
                evaluator = RAGEvaluator()
                result = evaluator.evaluate(samples)

            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Faithfulness", f"{result.faithfulness:.2%}")
            c2.metric("Answer Relevance", f"{result.answer_relevance:.2%}")
            c3.metric("Context Precision", f"{result.context_precision:.2%}")
            c4.metric("Context Recall", f"{result.context_recall:.2%}")
            c5.metric("Overall", f"{result.overall_score:.2%}")
