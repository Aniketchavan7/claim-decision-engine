"""LangGraph workflow definition for multi-agent claim analysis."""

from __future__ import annotations

import logging
from typing import Literal

from langgraph.graph import END, StateGraph

from src.agents.case_analysis import case_analysis_node
from src.agents.coverage_exclusion import coverage_exclusion_node
from src.agents.decision_agent import decision_node
from src.agents.policy_evidence import policy_evidence_node
from src.agents.state import AgentState
from src.agents.validation import validation_node
from src.config import get_settings

logger = logging.getLogger(__name__)


def route_after_validation(state: AgentState) -> Literal["coverage_exclusion", "__end__"]:
    """Conditional router after validation.

    If validation passes -> end workflow.
    If validation fails and retry_count < max_retries -> re-run Coverage & Exclusion.
    If retry limit exceeded -> force NEEDS_REVIEW and end workflow.
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
        "Max validation retries (%d) reached. Aborting retry loop; marked for review.",
        settings.max_validation_retries,
    )
    # Ensure decision reflects review requirement
    state["decision"] = "NEEDS_REVIEW"
    if "key_findings" in state:
        state["key_findings"].append(
            "System abstained from automated approval: validation could not confirm policy evidence support."
        )
    return END


def create_claim_workflow() -> StateGraph:
    """Construct and compile the LangGraph multi-agent claim analysis workflow."""
    workflow = StateGraph(AgentState)

    # Add agent nodes
    workflow.add_node("case_analysis", case_analysis_node)
    workflow.add_node("policy_evidence", policy_evidence_node)
    workflow.add_node("coverage_exclusion", coverage_exclusion_node)
    workflow.add_node("decision_agent", decision_node)
    workflow.add_node("validation_agent", validation_node)

    # Set workflow edges
    workflow.set_entry_point("case_analysis")
    workflow.add_edge("case_analysis", "policy_evidence")
    workflow.add_edge("policy_evidence", "coverage_exclusion")
    workflow.add_edge("coverage_exclusion", "decision_agent")
    workflow.add_edge("decision_agent", "validation_agent")

    # Conditional routing after validation (Pass vs Retry Loop)
    workflow.add_conditional_edges(
        "validation_agent",
        route_after_validation,
        {
            "coverage_exclusion": "coverage_exclusion",
            END: END,
        },
    )

    return workflow.compile()
