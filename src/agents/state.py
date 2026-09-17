"""Shared typed state exchanged between agents in the LangGraph workflow."""

from __future__ import annotations

from typing import Any, TypedDict

from src.models.claim import ClaimCase
from src.models.decision import (
    ApplicableLimit,
    Citation,
    ClaimDecision,
    ConfidenceBreakdown,
    DecisionStatus,
    Finding,
    TraceEntry,
    ValidationResult,
)
from src.models.evidence import DimensionEvidence, RankedEvidence


class InvestigationQuery(TypedDict):
    """A single investigation question with search queries."""

    dimension: str  # e.g. "waiting_period", "coverage", "exclusion"
    question: str  # Investigation question (NOT a conclusion)
    queries: list[str]  # Search queries for retrieval


class AgentState(TypedDict, total=False):
    """Full agent state passed through the LangGraph workflow.

    Using total=False so agents can write only their outputs without
    needing to provide every field.
    """

    # ── Input ────────────────────────────────────────────────────────
    claim_case: dict  # Raw claim case dict (serializable)

    # ── Case Analysis Agent output ───────────────────────────────────
    case_facts: dict  # Extracted + normalized facts
    decision_dimensions: list[str]  # What needs investigation
    investigation_plan: list[dict]  # list of InvestigationQuery dicts
    missing_fields: list[str]  # Fields needed but absent

    # ── Policy Evidence Agent output ─────────────────────────────────
    evidence_by_dimension: dict[str, dict]  # dimension → DimensionEvidence dict
    retrieval_metadata: dict[str, dict]  # Per-dimension stats

    # ── Coverage & Exclusion Agent output ────────────────────────────
    coverage_findings: list[dict]  # list of Finding dicts
    exclusion_findings: list[dict]
    waiting_period_assessment: dict
    applicable_limits: list[dict]  # list of ApplicableLimit dicts

    # ── Decision Agent output ────────────────────────────────────────
    decision: str  # DecisionStatus value
    confidence: float
    confidence_breakdown: dict  # ConfidenceBreakdown dict
    key_findings: list[str]
    missing_evidence: list[str]
    citations: list[dict]  # list of Citation dicts

    # ── Validation Agent output ──────────────────────────────────────
    validation: dict  # ValidationResult dict
    retry_count: int
    validation_feedback: list[str]

    # ── Trace ────────────────────────────────────────────────────────
    trace: list[dict]  # list of TraceEntry dicts
