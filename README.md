# 🔍 Multimodal RAG — Production-Grade AI System

> **Top-1% industry architecture** · YC-demo ready · FAANG system-design worthy · Open-source quality

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                         INGESTION PIPELINE                          │
│  PDF → [PdfTextParser + TableParser + ImageOcrParser]               │
│      → SemanticChunker (heading-aware, structure-preserving)        │
│      → HybridRetriever.index()                                      │
└──────────────────────────────┬──────────────────────────────────────┘
                               │ persisted index (FAISS + BM25 + pkl)
┌──────────────────────────────▼──────────────────────────────────────┐
│                          QUERY PIPELINE                             │
│                                                                     │
│  User Query                                                         │
│      │                                                              │
│      ▼                                                              │
│  LLMQueryProcessor  ─── rewrite + expand → sub-queries             │
│      │                                                              │
│      ▼                                                              │
│  HybridRetriever ──────── FAISS (dense) ──┐                         │
│  (multi-query)     └────── BM25  (sparse) ─┤─► RRF Fusion          │
│                                            │                         │
│      ▼                                                              │
│  CrossEncoderReranker  ── precision re-scoring                      │
│      │                                                              │
│      ▼                                                              │
│  AnswerGenerator ──────── GPT-4o-mini + citation-aware prompt      │
│      │                                                              │
│      ▼                                                              │
│  AnswerVerifier ────────── LLM self-check (hallucination filter)    │
│      │                                                              │
│      ▼                                                              │
│  Answer { text, citations, confidence, latency_ms, debug }          │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
multimodal-rag-pro/
│
├── config/
│   └── settings.py          # Pydantic Settings — all config from .env
│
├── core/
│   ├── models.py            # Domain models (Chunk, Answer, Citation …)
│   ├── interfaces.py        # Abstract base classes (swappable components)
│   └── rag_service.py       # Central orchestrator — the single entry point
│
├── ingestion/
│   ├── pipeline.py          # Orchestrates parse → chunk → index
│   ├── pdf_parser.py        # Layout-aware PyMuPDF text parser
│   ├── table_parser.py      # pdfplumber table → Markdown
│   └── image_processor.py   # Tesseract OCR with preprocessing
│
├── chunking/
│   └── semantic_chunker.py  # Heading-aware, structure-preserving chunker
│
├── embeddings/
│   ├── dense_encoder.py     # SentenceTransformer batched encoder
│   └── sparse_encoder.py    # BM25Okapi wrapper
│
├── retrieval/
│   ├── hybrid_retriever.py  # FAISS + BM25 + RRF fusion + persistence
│   ├── reranker.py          # Cross-encoder precision booster
│   └── query_processor.py   # LLM query rewriting + multi-query expansion
│
├── generation/
│   ├── answer_generator.py  # Citation-grounded GPT-4o-mini generation + streaming
│   └── verifier.py          # LLM hallucination self-check
│
├── evaluation/
│   └── evaluator.py         # RAGAS-style metrics (faithfulness, relevance …)
│
├── cache/
│   └── query_cache.py       # In-memory LRU cache with TTL
│
├── storage/
│   ├── redis_cache.py       # Production Redis cache (drop-in swap)
│   └── qdrant_store.py      # Qdrant vector store (drop-in swap for FAISS)
│
├── api/
│   ├── app.py               # FastAPI app factory + lifespan
│   ├── routers/
│   │   ├── ingest.py        # POST /ingest
│   │   ├── query.py         # POST /query, POST /query/stream
│   │   ├── evaluation.py    # POST /evaluate
│   │   └── feedback.py      # POST /feedback
│   └── schemas/
│       └── schemas.py       # Pydantic request/response models
│
├── ui/
│   └── app.py               # Streamlit UI (streaming, citations, diagnostics)
│
├── tests/
│   ├── unit/
│   │   ├── test_chunker.py
│   │   ├── test_retriever.py
│   │   └── test_parsers.py
│   └── integration/
│       └── test_pipeline.py
│
├── scripts/
│   ├── ingest.py            # CLI: ingest PDFs
│   └── query.py             # CLI: interactive query REPL
│
├── data/
│   ├── raw/                 # Source PDFs
│   ├── processed/           # Intermediate outputs
│   └── indexes/             # Persisted FAISS + BM25 indexes
│
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── pyproject.toml
└── .env.example
```

---

## Quickstart

### 1. Install

```bash
git clone <repo>
cd multimodal-rag-pro

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# System dependencies (macOS)
brew install tesseract poppler

# System dependencies (Ubuntu/Debian)
apt-get install -y tesseract-ocr poppler-utils
```

### 2. Configure

```bash
cp .env.example .env
# Edit .env and set OPENAI_API_KEY=sk-...
```

### 3a. Run the Streamlit UI

```bash
streamlit run ui/app.py
```
Open http://localhost:8501, upload a PDF, click **Build Index**, then ask questions.

### 3b. Run the FastAPI Backend

```bash
uvicorn api.app:app --reload
```
Docs at http://localhost:8000/docs

### 3c. CLI Tools

```bash
# Ingest
python scripts/ingest.py data/raw/my_doc.pdf

# Query REPL
python scripts/query.py
```

---

## Docker

```bash
# Build and run both API + UI
docker-compose up --build

# API: http://localhost:8000
# UI:  http://localhost:8501
```

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/ingest` | Upload PDF (multipart/form-data) |
| `POST` | `/query` | Synchronous Q&A |
| `POST` | `/query/stream` | Streaming Q&A (SSE) |
| `POST` | `/feedback` | Thumbs up/down signal |
| `POST` | `/evaluate` | Run RAGAS evaluation |
| `GET` | `/health` | Liveness probe |
| `GET` | `/stats` | Index + cache statistics |

### Example: Query

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "text": "What is Qatar GDP in 2022?",
    "top_k": 5,
    "enable_reranking": true,
    "enable_query_rewriting": true
  }'
```

---

## Key Technical Decisions

### Hybrid Retrieval (BM25 + Dense + RRF)
Pure dense retrieval fails on exact keyword lookups (names, codes, IDs). BM25 captures term frequency. Reciprocal Rank Fusion merges them without any calibration — more robust than linear interpolation.

### Semantic Chunking
Splitting at heading boundaries avoids context fragmentation. Tables are never split (mid-table splits destroy reasoning ability). Each chunk carries `section_heading` metadata for provenance.

### Cross-Encoder Reranking
Bi-encoders (used for recall) compute query and passage embeddings independently — fast but imprecise. Cross-encoders attend to both jointly — much more accurate. Used only on the short shortlist (top-20 → top-5) so latency stays acceptable.

### Hallucination Verification
A second LLM call checks each claim in the answer against the retrieved chunks. Adds ~300ms but catches the worst confabulations before they reach the user.

### Index Persistence
FAISS index + chunk metadata pickled to disk after every ingest. On startup the API reloads it — zero cold-start re-embedding. For millions of documents, swap to Qdrant (see `storage/qdrant_store.py`).

---

## Production Scaling Guide

| Concern | Current | Production Swap |
|---------|---------|-----------------|
| Vector store | FAISS (in-process) | Qdrant Cloud / Pinecone |
| Cache | In-memory LRU | Redis (`storage/redis_cache.py`) |
| Embedding service | In-process ST | Dedicated FastAPI embedding service |
| Job queue | Synchronous | Celery + Redis / AWS SQS |
| Horizontal scaling | Single process | Docker + K8s (stateless API + shared Qdrant) |
| Observability | Logging | OpenTelemetry + Jaeger / Datadog |

---

## Evaluation

Run RAGAS-style automated evaluation via UI or API:

```bash
curl -X POST http://localhost:8000/evaluate \
  -H "Content-Type: application/json" \
  -d '{
    "samples": [
      {
        "question": "What is Qatar GDP?",
        "ground_truth": "Qatar GDP was $179B in 2021.",
        "predicted_answer": "...",
        "retrieved_contexts": ["..."]
      }
    ]
  }'
```

Metrics returned: `faithfulness`, `answer_relevance`, `context_precision`, `context_recall`, `overall_score`.

---

## Testing

```bash
# Unit tests only (no API key needed)
pytest tests/unit/ -v

# Integration tests (requires sample PDF)
pytest tests/integration/ -v

# All tests
pytest -v
```

---

## Roadmap / Wow-Factor Extensions

- [ ] **Agentic multi-hop reasoning** — ReAct-style loop for multi-step retrieval
- [ ] **RLHF feedback loop** — collect thumbs signals, fine-tune retrieval weights
- [ ] **Document highlighting** — return bounding-box coordinates for citation highlighting
- [ ] **Vision captioning** — GPT-4o image understanding (not just OCR text)
- [ ] **Async ingestion queue** — Celery workers for large-scale batch ingestion
- [ ] **Pinecone / Weaviate adapters** — beyond Qdrant
- [ ] **Observability dashboard** — latency histograms, cache hit rate, retrieval quality over time

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
