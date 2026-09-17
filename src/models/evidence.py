"""Pydantic models for policy evidence, chunks, and retrieval results."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class PolicyChunk(BaseModel):
    """A single chunk of policy text with full provenance metadata."""

    chunk_id: str
    text: str
    page: int
    section: str = ""
    subsection: str = ""
    heading_path: str = ""  # e.g. "Scope of Cover > Hospitalization Benefits"
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RetrievedEvidence(BaseModel):
    """Result from a single retrieval method (dense or sparse)."""

    chunk: PolicyChunk
    score: float = 0.0
    retrieval_method: str = "dense"  # "dense" or "sparse"

    # Convenience properties for backward compatibility
    @property
    def chunk_id(self) -> str:
        return self.chunk.chunk_id

    @property
    def text(self) -> str:
        return self.chunk.text

    @property
    def metadata(self) -> Dict[str, Any]:
        return self.chunk.metadata


class RankedEvidence(BaseModel):
    """Result after RRF fusion and/or reranking."""

    chunk: PolicyChunk
    final_score: float = 0.0
    dense_score: float = 0.0
    sparse_score: float = 0.0
    rerank_score: float = 0.0
    dense_rank: int = 0
    sparse_rank: int = 0
    retrieval_method: str = "hybrid_rrf"

    # Convenience properties for backward compatibility
    @property
    def chunk_id(self) -> str:
        return self.chunk.chunk_id

    @property
    def text(self) -> str:
        return self.chunk.text

    @property
    def metadata(self) -> Dict[str, Any]:
        return self.chunk.metadata

    @property
    def score(self) -> float:
        return self.final_score

    @score.setter
    def score(self, value: float) -> None:
        self.final_score = value


class DimensionEvidence(BaseModel):
    """Evidence grouped by a single investigation dimension."""

    dimension: str  # e.g. "waiting_period", "coverage", "exclusion"
    queries_used: List[str] = Field(default_factory=list)
    results: List[RankedEvidence] = Field(default_factory=list)
    retrieval_stats: Dict[str, Any] = Field(default_factory=dict)

    # Alias / backward compatibility
    @property
    def evidence(self) -> List[RankedEvidence]:
        return self.results
