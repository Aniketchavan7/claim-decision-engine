"""LangGraph workflow definition for multi-agent claim analysis."""

from __future__ import annotations

import logging
from typing import Any, Literal

from langgraph.graph import END, StateGraph

from src.agents.case_analysis import case_analysis_node
from src.agents.coverage_exclusion import coverage_exclusion_node
from src.agents.decision_agent import decision_node
from src.agents.policy_evidence import policy_evidence_node
from src.agents.state import AgentState
from src.agents.validation import validation_node
from src.config import get_settings

logger = logging.getLogger(__name__)


def force_needs_review_node(state: AgentState) -> dict[str, Any]:
    """Exhausted retries safety node: explicitly updates state to NEEDS_REVIEW with audit reasons.
    
    This ensures that when validation fails repeatedly, the final decision is deterministically
    forced to NEEDS_REVIEW via a proper LangGraph node rather than a dead mutation inside a router.
    """
    validation = state.get("validation", {})
    unsupported = validation.get("unsupported_claims", [])
    feedback = validation.get("feedback", [])
    retry_count = state.get("retry_count", 0)

    logger.warning(
        "force_needs_review_node: max validation retries (%d) reached with unconfirmed evidence. Enforcing NEEDS_REVIEW.",
        retry_count,
    )

    abstention_reasons = [
        "System abstained from automated approval: validation could not confirm policy evidence support after retries."
    ]
    if unsupported:
        abstention_reasons.append(f"Unverified evidence claims: {'; '.join(str(u) for u in unsupported[:3])}")
    if feedback:
        abstention_reasons.append(f"Validation auditor feedback: {'; '.join(str(fb) for fb in feedback[:2])}")

    key_findings = list(state.get("key_findings", []))
    for r in abstention_reasons:
        if r not in key_findings:
            key_findings.append(r)

    missing_evidence = list(state.get("missing_evidence", []))
    for r in abstention_reasons:
        if r not in missing_evidence:
            missing_evidence.append(r)

    cb = dict(state.get("confidence_breakdown", {}))
    cb["missing_critical"] = 0.0
    cb["validation_result"] = 0.0

    trace_entry = {
        "agent_name": "Safety Fallback (Force NEEDS_REVIEW)",
        "action": f"Validation unconfirmed after {retry_count} retries; enforced safety abstention",
        "duration_ms": 1,
        "retrieval_count": 0,
        "status": "completed",
        "details": {
            "prior_decision": state.get("decision", ""),
            "unsupported_claims_count": len(unsupported),
            "unsupported_claims": unsupported[:3],
        },
    }

    return {
        "decision": "NEEDS_REVIEW",
        "confidence": round(min(state.get("confidence", 0.5), 0.65), 3),
        "confidence_breakdown": cb,
        "key_findings": key_findings,
        "missing_evidence": missing_evidence,
        "trace": state.get("trace", []) + [trace_entry],
    }


def route_after_validation(state: AgentState) -> Literal["coverage_exclusion", "force_needs_review", "__end__"]:
    """Conditional router after validation.

    If validation passes -> end workflow.
    If validation fails and retry_count <= max_retries -> re-run Coverage & Exclusion.
    If retry limit exceeded -> route to force_needs_review node for explicit state update.
    """
    settings = get_settings()
    validation = state.get("validation", {})
    status = validation.get("status", "PASS")
    decision = state.get("decision", "")
    retry_count = state.get("retry_count", 0)

    # When the system legitimately abstains (NEEDS_REVIEW), do not loop in retry
    if decision == "NEEDS_REVIEW":
        logger.info("Decision is NEEDS_REVIEW (Abstention). Validation complete, ending workflow.")
        return END

    if status == "PASS":
        logger.info("Workflow validation PASSED. Proceeding to final output.")
        return END

    if retry_count <= settings.max_validation_retries:
        logger.warning(
            "Workflow validation FAILED (attempt %d/%d). Routing back to coverage_exclusion for refinement.",
            retry_count,
            settings.max_validation_retries,
        )
        return "coverage_exclusion"

    logger.warning(
        "Max validation retries (%d) reached. Routing to force_needs_review node.",
        settings.max_validation_retries,
    )
    return "force_needs_review"


def create_claim_workflow() -> StateGraph:
    """Construct and compile the LangGraph multi-agent claim analysis workflow."""
    workflow = StateGraph(AgentState)

    # Add agent nodes
    workflow.add_node("case_analysis", case_analysis_node)
    workflow.add_node("policy_evidence", policy_evidence_node)
    workflow.add_node("coverage_exclusion", coverage_exclusion_node)
    workflow.add_node("decision_agent", decision_node)
    workflow.add_node("validation_agent", validation_node)
    workflow.add_node("force_needs_review", force_needs_review_node)

    # Set workflow edges
    workflow.set_entry_point("case_analysis")
    workflow.add_edge("case_analysis", "policy_evidence")
    workflow.add_edge("policy_evidence", "coverage_exclusion")
    workflow.add_edge("coverage_exclusion", "decision_agent")
    workflow.add_edge("decision_agent", "validation_agent")
    workflow.add_edge("force_needs_review", END)

    # Conditional routing after validation (Pass vs Retry Loop vs Forced Review)
    workflow.add_conditional_edges(
        "validation_agent",
        route_after_validation,
        {
            "coverage_exclusion": "coverage_exclusion",
            "force_needs_review": "force_needs_review",
            END: END,
        },
    )

    return workflow.compile()
