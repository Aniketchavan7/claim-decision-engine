import pytest
from src.agents.state import AgentState
from src.workflow.graph import force_needs_review_node, route_after_validation
from langgraph.graph import END


def test_validation_routing_pass():
    state: AgentState = {
        "validation": {"status": "PASS"},
        "retry_count": 0,
    }
    next_node = route_after_validation(state)
    assert next_node == END


def test_validation_routing_retry():
    state: AgentState = {
        "validation": {"status": "FAIL", "unsupported_claims": ["hallucinated limit"]},
        "retry_count": 0,
    }
    next_node = route_after_validation(state)
    assert next_node == "coverage_exclusion"


def test_validation_routing_max_retries():
    state: AgentState = {
        "validation": {"status": "FAIL", "unsupported_claims": ["hallucinated limit"]},
        "retry_count": 3,
    }
    next_node = route_after_validation(state)
    assert next_node == "force_needs_review"


def test_force_needs_review_node():
    state: AgentState = {
        "decision": "ADMISSIBLE_WITH_LIMITS",
        "confidence": 0.95,
        "validation": {
            "status": "FAIL",
            "unsupported_claims": ["hallucinated limit on room rent"],
            "feedback": ["Room rent deduction unverified by cited chunk."],
        },
        "key_findings": ["Inpatient surgery covered."],
        "missing_evidence": [],
        "confidence_breakdown": {
            "evidence_support": 0.9,
            "missing_critical": 1.0,
            "validation_result": 0.5,
        },
        "retry_count": 2,
        "trace": [],
    }

    result = force_needs_review_node(state)

    # 1. State update explicitly forces NEEDS_REVIEW
    assert result["decision"] == "NEEDS_REVIEW"
    # 2. Confidence is penalized
    assert result["confidence"] <= 0.65
    assert result["confidence_breakdown"]["missing_critical"] == 0.0
    assert result["confidence_breakdown"]["validation_result"] == 0.0
    # 3. Unsupported claims and feedback are recorded in key_findings and missing_evidence
    assert any("hallucinated limit on room rent" in f for f in result["key_findings"])
    assert any("Room rent deduction unverified" in m for m in result["missing_evidence"])
    # 4. Audit trace entry recorded
    assert len(result["trace"]) == 1
    assert result["trace"][0]["agent_name"] == "Safety Fallback (Force NEEDS_REVIEW)"
