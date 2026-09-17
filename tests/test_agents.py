import pytest
from src.agents.state import AgentState
from src.workflow.graph import route_after_validation
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
    assert next_node == END
    assert state.get("decision") == "NEEDS_REVIEW"
