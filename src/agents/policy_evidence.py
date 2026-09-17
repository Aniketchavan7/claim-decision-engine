"""Policy Evidence Agent — dimension-specific retrieval orchestrator.

This agent does NOT use an LLM. It is a pure retrieval orchestrator
that runs the hybrid retrieval pipeline independently for each
investigation dimension.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from src.agents.state import AgentState

logger = logging.getLogger(__name__)

# Global retriever instance — set by the workflow at startup
_hybrid_retriever = None


def set_retriever(retriever: Any) -> None:
    """Set the global hybrid retriever instance."""
    global _hybrid_retriever
    _hybrid_retriever = retriever


def policy_evidence_node(state: AgentState) -> dict[str, Any]:
    """LangGraph node: run dimension-specific retrieval for each investigation query."""
    t0 = time.time()
    logger.info("Policy Evidence Agent: starting retrieval")

    if _hybrid_retriever is None:
        raise RuntimeError(
            "Hybrid retriever not initialized. Call set_retriever() first."
        )

    investigation_plan = state.get("investigation_plan", [])

    # Build dimensions dict: {dimension_name: [queries]}
    dimensions: dict[str, list[str]] = {}
    for item in investigation_plan:
        dim = item.get("dimension", "general")
        queries = item.get("queries", [])
        if dim in dimensions:
            dimensions[dim].extend(queries)
        else:
            dimensions[dim] = list(queries)

    # Run dimension-specific retrieval
    evidence_by_dimension = _hybrid_retriever.retrieve_by_dimensions(dimensions)

    # Convert DimensionEvidence objects to dicts for state serialization
    evidence_dicts = {}
    retrieval_metadata = {}
    total_results = 0

    for dim_name, dim_evidence in evidence_by_dimension.items():
        evidence_dicts[dim_name] = dim_evidence.model_dump()
        total_results += len(dim_evidence.results)
        retrieval_metadata[dim_name] = {
            "result_count": len(dim_evidence.results),
            "queries_used": len(dim_evidence.queries_used),
            "top_score": (
                max(r.final_score for r in dim_evidence.results)
                if dim_evidence.results
                else 0.0
            ),
        }

    duration_ms = int((time.time() - t0) * 1000)
    logger.info(
        "Policy Evidence Agent: retrieved %d total results across %d dimensions in %dms",
        total_results,
        len(dimensions),
        duration_ms,
    )

    trace_entry = {
        "agent_name": "Policy Evidence Agent",
        "action": f"Retrieved evidence for {len(dimensions)} dimensions",
        "duration_ms": duration_ms,
        "retrieval_count": total_results,
        "status": "completed",
        "details": retrieval_metadata,
    }

    return {
        "evidence_by_dimension": evidence_dicts,
        "retrieval_metadata": retrieval_metadata,
        "trace": state.get("trace", []) + [trace_entry],
    }
