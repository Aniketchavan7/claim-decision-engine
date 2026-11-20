"""Unit tests for insurance claim decision Pydantic data schemas."""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import pytest
from models.decision import (
    DecisionStatus,
    Citation,
    Finding,
    ApplicableLimit,
    ConfidenceBreakdown,
    ClaimDecision,
    ValidationResult,
)


def test_decision_status_enum():
    """Verify all valid decision statuses exist."""
    assert DecisionStatus.ADMISSIBLE == "ADMISSIBLE"
    assert DecisionStatus.NOT_ADMISSIBLE == "NOT_ADMISSIBLE"
    assert DecisionStatus.PARTIALLY_ADMISSIBLE == "PARTIALLY_ADMISSIBLE"


def test_confidence_composite_score():
    """Verify weighted composite confidence score calculation."""
    breakdown = ConfidenceBreakdown(
        evidence_support=1.0,
        citation_coverage=0.8,
        retrieval_quality=0.7,
        validation_result=1.0,
        missing_critical=1.0,
    )
    # Expected: 0.30*1.0 + 0.25*0.8 + 0.15*0.7 + 0.20*1.0 + 0.10*1.0
    # = 0.30 + 0.20 + 0.105 + 0.20 + 0.10 = 0.905
    assert breakdown.composite_score == 0.905


def test_claim_decision_serialization():
    """Verify ClaimDecision serializes and validates correctly."""
    decision = ClaimDecision(
        case_id="CLM-2026-001",
        decision=DecisionStatus.ADMISSIBLE_WITH_LIMITS,
        confidence=0.88,
        key_findings=["In-network hospital procedure", "Room rent cap applied"],
        applicable_limits=[
            ApplicableLimit(
                limit_type="room_rent",
                description="Daily room rent capped at 1% of Sum Insured",
                limit_amount=5000.0,
                claimed_amount=8000.0,
                payable_amount=5000.0,
                policy_reference="Clause 3.1",
            )
        ],
        citations=[
            Citation(
                claim="Room rent capped",
                page=14,
                section="3.1",
                chunk_id="chunk_14_01",
                chunk_text="Room rent shall not exceed 1% of Sum Insured per day.",
            )
        ],
    )
    assert decision.case_id == "CLM-2026-001"
    assert decision.decision == DecisionStatus.ADMISSIBLE_WITH_LIMITS
    assert len(decision.applicable_limits) == 1
    assert decision.applicable_limits[0].payable_amount == 5000.0
    assert len(decision.citations) == 1
