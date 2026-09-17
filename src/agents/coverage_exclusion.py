"""Coverage & Exclusion Agent — policy reasoning specialist.

Takes case facts + dimension-grouped evidence and reasons about each
dimension using the actual policy text. Every finding MUST be backed
by a specific policy citation.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from src.agents.state import AgentState
from src.llm.client import get_reasoning_client

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a Coverage & Exclusion Agent for health insurance claims.

You receive:
1. Case facts (patient, treatment, hospital, expenses, etc.)
2. Policy evidence grouped by investigation dimension (actual policy text chunks)

Your job is to reason about EACH dimension using ONLY the provided policy evidence.

CRITICAL RULES:
- Base every finding on the ACTUAL policy text provided. Quote or reference it.
- If the evidence is insufficient to make a determination, say "INSUFFICIENT_EVIDENCE" for that dimension.
- Do NOT invent policy rules from general insurance knowledge.
- Do NOT make assumptions about what the policy says if the evidence doesn't cover it.
- Each finding must reference the chunk_id(s) that support it.

For each dimension, assess:
- COVERAGE: Is the treatment/condition covered? What benefits apply?
- WAITING_PERIOD: Does any waiting period (initial 30-day, 1-year specific disease under Clause 3, 4-year PED) block the claim?
- EXCLUSION: Is the treatment explicitly excluded (cosmetic, experimental, specific conditions)?
- HOSPITAL_DEFINITION: Does the facility meet the policy's hospital definition?
- LIMITS: What sub-limits, caps, or per-day limits apply (room rent, ambulance, procedure-specific)?
- PRE_POST_HOSPITALIZATION: Do expenses fall within allowed time windows (typically 30 days before, 60 days after)?
- PORTABILITY: Does prior continuous coverage reduce waiting periods?
- DOMICILIARY_CONDITIONS: Does domiciliary treatment meet all policy conditions?

Respond with JSON:
{
  "coverage_findings": [
    {
      "description": "...",
      "category": "coverage",
      "dimension": "coverage",
      "supported": true,
      "evidence_references": ["chunk_id_1", "chunk_id_2"]
    }
  ],
  "exclusion_findings": [
    {
      "description": "Cosmetic surgery is explicitly excluded under General Exclusions",
      "category": "exclusion",
      "dimension": "exclusion",
      "supported": true,
      "evidence_references": ["chunk_5_012"]
    }
  ],
  "waiting_period_assessment": {
    "initial_waiting_period_applies": false,
    "specific_disease_waiting_applies": false,
    "ped_waiting_applies": false,
    "waiting_period_satisfied": true,
    "details": "...",
    "evidence_references": ["chunk_id"]
  },
  "applicable_limits": [
    {
      "limit_type": "room_rent",
      "description": "Room rent limited to 1% of sum insured per day",
      "limit_amount": 5000,
      "claimed_amount": 30000,
      "payable_amount": 20000,
      "policy_reference": "chunk_id"
    }
  ]
}
"""

RETRY_ADDENDUM = """
IMPORTANT: The Validation Agent found that the following claims were NOT supported by policy evidence:
{unsupported_claims}

Please re-examine these specific points. Either:
1. Find better evidence from the provided chunks to support the claim
2. Revise the finding to match what the evidence actually says
3. Mark as INSUFFICIENT_EVIDENCE if the policy doesn't address this

Be more careful about citing evidence accurately this time.
"""


def coverage_exclusion_node(state: AgentState) -> dict[str, Any]:
    """LangGraph node: reason about coverage, exclusions, limits using policy evidence."""
    t0 = time.time()
    retry_count = state.get("retry_count", 0)
    logger.info("Coverage & Exclusion Agent: starting (retry=%d)", retry_count)

    case_facts = state.get("case_facts", {})
    evidence_by_dimension = state.get("evidence_by_dimension", {})
    validation_feedback = state.get("validation_feedback", [])

    # Build the evidence context for the LLM
    evidence_text = _format_evidence(evidence_by_dimension)

    user_msg = f"""Analyze this claim against the provided policy evidence.

CASE FACTS:
{_format_dict(case_facts)}

POLICY EVIDENCE BY DIMENSION:
{evidence_text}

Provide your structured findings as JSON."""

    # Add retry context if this is a retry
    system = SYSTEM_PROMPT
    if retry_count > 0 and validation_feedback:
        system += RETRY_ADDENDUM.format(
            unsupported_claims="\n".join(f"- {c}" for c in validation_feedback)
        )

    client = get_reasoning_client()
    result = client.generate_json(
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user_msg},
        ],
        max_tokens=4096,
    )

    duration_ms = int((time.time() - t0) * 1000)
    n_coverage = len(result.get("coverage_findings", []))
    n_exclusion = len(result.get("exclusion_findings", []))
    n_limits = len(result.get("applicable_limits", []))

    logger.info(
        "Coverage & Exclusion Agent: %d coverage, %d exclusion, %d limits in %dms",
        n_coverage, n_exclusion, n_limits, duration_ms,
    )

    trace_entry = {
        "agent_name": "Coverage & Exclusion Agent",
        "action": f"Analyzed {n_coverage} coverage, {n_exclusion} exclusion findings, {n_limits} limits",
        "duration_ms": duration_ms,
        "retrieval_count": 0,
        "status": "completed" if retry_count == 0 else f"retry_{retry_count}",
        "details": {
            "coverage_count": n_coverage,
            "exclusion_count": n_exclusion,
            "limits_count": n_limits,
            "is_retry": retry_count > 0,
        },
    }

    return {
        "coverage_findings": result.get("coverage_findings", []),
        "exclusion_findings": result.get("exclusion_findings", []),
        "waiting_period_assessment": result.get("waiting_period_assessment", {}),
        "applicable_limits": result.get("applicable_limits", []),
        "trace": state.get("trace", []) + [trace_entry],
    }


def _format_evidence(evidence_by_dimension: dict) -> str:
    """Format evidence for the LLM prompt."""
    parts = []
    for dim, dim_data in evidence_by_dimension.items():
        parts.append(f"\n--- DIMENSION: {dim.upper()} ---")
        results = dim_data.get("results", [])
        if not results:
            parts.append("  No evidence found for this dimension.")
            continue
        for i, r in enumerate(results[:5], 1):  # Top 5 most relevant chunks per dimension
            chunk = r.get("chunk", {})
            parts.append(
                f"  [{i}] chunk_id={chunk.get('chunk_id', '?')} | "
                f"page={chunk.get('page', '?')} | "
                f"section={chunk.get('section', '?')} | "
                f"score={r.get('final_score', 0):.3f}\n"
                f"  {chunk.get('text', '')[:450]}"
            )
    return "\n".join(parts)


def _format_dict(d: dict) -> str:
    import json
    return json.dumps(d, indent=2, default=str)
