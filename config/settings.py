"""
Centralised configuration — loaded from environment / .env file.
Never hard-code secrets.  All tunables live here.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ------------------------------------------------------------------ LLM
    gemini_api_key: str = Field(default="", alias="GEMINI_API_KEY")
    llm_model: str = "gemini-2.5-flash"
    llm_temperature: float = 0.0
    llm_max_tokens: int = 1024
    llm_timeout: int = 30
    llm_fallback_model: str = "gemini-2.0-flash"  # used by verifier + on 503

    # ------------------------------------------------------------ Embeddings
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_batch_size: int = 64
    embedding_cache_dir: str = ".cache/embeddings"

    # --------------------------------------------------------------- Chunking
    chunk_size: int = 512
    chunk_overlap: int = 64
    min_chunk_length: int = 30

    # --------------------------------------------------------------- Retrieval
    default_top_k: int = 5
    bm25_top_k: int = 10
    dense_top_k: int = 10
    hybrid_alpha: float = 0.5          # weight for dense (1-alpha = sparse)
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    reranker_top_k: int = 5

    # ---------------------------------------------------------------- Storage
    data_dir: str = "data"
    index_dir: str = "data/indexes"
    raw_dir: str = "data/raw"
    processed_dir: str = "data/processed"

    # ------------------------------------------------------------------ Cache
    cache_enabled: bool = True
    cache_ttl: int = 3600          # seconds
    query_cache_max_size: int = 256

    # ------------------------------------------------------------------- API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_workers: int = 1
    api_reload: bool = False
    log_level: str = "INFO"

    # ---------------------------------------------------------------- Features
    enable_query_rewriting: bool = True
    enable_reranking: bool = True
    enable_verification: bool = True
    enable_ocr: bool = True

    # ---------------------------------------------------------- OCR / Vision
    ocr_dpi: int = 300
    ocr_lang: str = "eng"
    tesseract_config: str = "--oem 3 --psm 6"

    @property
    def index_path(self) -> Path:
        return Path(self.index_dir)

    @property
    def raw_path(self) -> Path:
        return Path(self.raw_dir)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
