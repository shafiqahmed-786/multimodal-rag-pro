"""
/ingest router — upload and index documents.
"""
from __future__ import annotations

import logging
import shutil
import tempfile
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request, UploadFile, File

from api.schemas.schemas import IngestResponse
from config.settings import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()
router = APIRouter()

_ALLOWED_EXTENSIONS = {".pdf"}


@router.post("", response_model=IngestResponse)
async def ingest_document(
    request: Request,
    file: UploadFile = File(...),
):
    """Upload a PDF and add it to the retrieval index."""
    ext = Path(file.filename or "").suffix.lower()
    if ext not in _ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{ext}'. Allowed: {_ALLOWED_EXTENSIONS}",
        )

    rag = request.app.state.rag

    # Save upload to a temp file so parsers can access it by path
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        result = rag.ingest(tmp_path)
        # Persist updated index
        rag.save_index()
    except Exception as exc:
        logger.exception("Ingestion failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    return IngestResponse(
        file=file.filename or "unknown",
        chunks_indexed=result["chunks_indexed"],
    )
