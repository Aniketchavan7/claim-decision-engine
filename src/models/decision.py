"""Pydantic models for the structured decision output contract."""

from __future__ import annotations

from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class DecisionStatus(str, Enum):
    ADMISSIBLE = "ADMISSIBLE"
    ADMISSIBLE_WITH_LIMITS = "ADMISSIBLE_WITH_LIMITS"
    PARTIALLY_ADMISSIBLE = "PARTIALLY_ADMISSIBLE"
    NOT_ADMISSIBLE = "NOT_ADMISSIBLE"
    NEEDS_REVIEW = "NEEDS_REVIEW"


class Citation(BaseModel):
    """A traceable policy citation backing a decision claim."""

    claim: str  # The statement being supported
    source: str = "policy.pdf"
    page: int = 0
    section: str = ""
    chunk_id: str = ""
    chunk_text: str = ""  # The actual policy text cited


class Finding(BaseModel):
    """A structured finding from the Coverage & Exclusion Agent."""

    description: str
    category: str = ""  # coverage, exclusion, waiting_period, limits, etc.
    dimension: str = ""
    supported: bool = True
    evidence_references: list[str] = Field(default_factory=list)  # chunk_ids


class ApplicableLimit(BaseModel):
    """A policy limit, cap, or deduction that affects the payable amount."""

    limit_type: str  # room_rent, ambulance, sub_limit, etc.
    description: str
    limit_amount: Optional[float] = None
    claimed_amount: Optional[float] = None
    payable_amount: Optional[float] = None
    policy_reference: str = ""


class ValidationResult(BaseModel):
    """Outcome of the Validation Agent's quality check."""

    status: str = "PENDING"  # PASS or FAIL
    unsupported_claims: list[str] = Field(default_factory=list)
    feedback: list[str] = Field(default_factory=list)


class ConfidenceBreakdown(BaseModel):
    """Evidence-based confidence — NOT a calibrated probability."""

    evidence_support: float = 0.0  # fraction of dimensions with strong evidence
    citation_coverage: float = 0.0  # fraction of findings with policy citations
    retrieval_quality: float = 0.0  # avg reranker score of top evidence
    validation_result: float = 0.0  # 1.0=PASS, 0.5=retry-PASS, 0.0=FAIL
    missing_critical: float = 1.0  # 1.0=no missing critical, 0.0=critical missing

    @property
    def composite_score(self) -> float:
        weights = [0.30, 0.25, 0.15, 0.20, 0.10]
        values = [
            self.evidence_support,
            self.citation_coverage,
            self.retrieval_quality,
            self.validation_result,
            self.missing_critical,
        ]
        return round(sum(w * v for w, v in zip(weights, values)), 3)


class TraceEntry(BaseModel):
    """A single entry in the execution trace (visible, no hidden CoT)."""

    agent_name: str
    action: str
    duration_ms: int = 0
    retrieval_count: int = 0
    status: str = ""
    details: dict = Field(default_factory=dict)


class ClaimDecision(BaseModel):
    """The final structured decision response returned by the API."""

    case_id: str
    decision: DecisionStatus
    confidence: float = 0.0
    confidence_breakdown: ConfidenceBreakdown = Field(
        default_factory=ConfidenceBreakdown
    )
    key_findings: list[str] = Field(default_factory=list)
    applicable_limits: list[ApplicableLimit] = Field(default_factory=list)
    missing_evidence: list[str] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    validation: ValidationResult = Field(default_factory=ValidationResult)
    trace: list[TraceEntry] = Field(default_factory=list)
