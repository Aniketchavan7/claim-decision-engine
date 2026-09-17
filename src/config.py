"""Application configuration using pydantic-settings."""

from __future__ import annotations

import os
from pathlib import Path
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central configuration — all values overridable via environment variables."""

    # ── LLM (provider-agnostic) ──────────────────────────────────────
    llm_provider: str = "xkiro"
    llm_api_key: str = ""
    llm_reasoning_model: str = "qwen/qwen3.8-max:free"
    llm_fast_model: str = "qwen/qwen3.7-flash:free"
    llm_fallback_model: str = "minimax/minimax-m3:free"
    llm_temperature: float = 0.1

    # ── Embedding (local) ────────────────────────────────────────────
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_query_prefix: str = (
        "Represent this sentence for searching relevant passages: "
    )

    # ── Reranker (local) ─────────────────────────────────────────────
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # ── Retrieval knobs ──────────────────────────────────────────────
    dense_top_k: int = 25
    sparse_top_k: int = 25
    fusion_top_k: int = 15
    rerank_top_k: int = 10
    rrf_k: int = 60

    # ── Chunking ─────────────────────────────────────────────────────
    max_chunk_tokens: int = 800
    chunk_overlap_tokens: int = 100

    # ── Workflow ─────────────────────────────────────────────────────
    max_validation_retries: int = 2
    analysis_timeout_seconds: int = 240

    # ── Paths ────────────────────────────────────────────────────────
    policy_pdf_path: str = "policy/USGIC-CSCIndividualHealthInsurance_2017-2018.pdf"
    index_dir: str = "data/index"
    project_root: str = ""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    def resolve_path(self, relative: str) -> Path:
        """Resolve a path relative to the project root."""
        root = Path(self.project_root) if self.project_root else Path.cwd()
        return root / relative


# Provider base URLs for the OpenAI-compatible SDK
PROVIDER_URLS: dict[str, str] = {
    "xkiro": "https://api.xkiro.com/v1",
    "openai": "https://api.openai.com/v1",
    "groq": "https://api.groq.com/openai/v1",
    "gemini": "https://generativelanguage.googleapis.com/v1beta/openai",
}


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached singleton of the application settings."""
    return Settings()
