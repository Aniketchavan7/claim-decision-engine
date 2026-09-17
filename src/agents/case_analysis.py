"""Case Analysis Agent — investigation planner (NOT policy interpreter).

This agent extracts facts and generates investigation questions.
It NEVER concludes policy applicability — it only asks questions.

✅ "Investigate whether a specific disease waiting period applies to thyroid disorder"
❌ "This claim violates the 2-year waiting period for thyroid"
"""

from __future__ import annotations

import logging
import time
from typing import Any

from src.agents.state import AgentState
from src.llm.client import get_fast_client

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a Case Analysis Agent for health insurance claims.

Your job is to:
1. Extract and normalize the key facts from the claim case
2. Identify which DIMENSIONS need investigation against the policy
3. For each dimension, generate INVESTIGATION QUESTIONS (not conclusions)
4. Generate specific SEARCH QUERIES to find relevant policy clauses
5. Flag any missing or insufficient evidence

CRITICAL RULES:
- You NEVER make policy conclusions. You only ask what needs to be investigated.
- You do NOT say "this is excluded" or "this violates waiting period" — you say "investigate whether X applies"
- Each dimension should have 2-4 specific search queries optimized for retrieval

Respond with a JSON object with this exact structure:
{
  "case_facts": {
    "case_id": "...",
    "patient_age": ...,
    "treatment_type": "...",
    "diagnosis": "...",
    "procedure": "...",
    "is_pre_existing": ...,
    "is_experimental": ...,
    "admission_hours": ...,
    "hospital_name": "...",
    "is_network_hospital": ...,
    "sum_insured": ...,
    "policy_start_date": "...",
    "claim_date": "...",
    "continuous_coverage_months": ...,
    "prior_insurer_years": ...,
    "total_claimed": ...,
    "documents_available": [...],
    "has_prior_policy": ...,
    "has_expense_timing": ...,
    "has_evidence_context": ...
  },
  "decision_dimensions": ["coverage", "waiting_period", ...],
  "investigation_plan": [
    {
      "dimension": "coverage",
      "question": "Investigate whether inpatient hospitalization for acute appendicitis is covered under the policy",
      "queries": [
        "inpatient hospitalization coverage benefits scope",
        "appendectomy surgical procedure coverage",
        "hospitalization benefits covered expenses"
      ]
    },
    ...
  ],
  "missing_fields": ["list of any critical missing information"]
}

DIMENSION TYPES to consider (select only those relevant to the case):
- "coverage": Is the treatment type (inpatient/day_care/domiciliary) covered?
- "waiting_period": Does any waiting period (initial 30-day, specific disease, PED) affect this claim?
- "exclusion": Is the treatment/condition excluded (cosmetic, experimental, etc.)?
- "hospital_definition": Does the facility meet the policy definition of a hospital?
- "limits": Are there room rent, ambulance, or category-specific sub-limits?
- "pre_post_hospitalization": Do pre/post hospitalization expenses fall within allowed time windows?
- "portability": Does prior continuous coverage affect waiting periods?
- "medical_necessity": Is there evidence of medical necessity?
- "domiciliary_conditions": Does domiciliary treatment meet policy conditions?
"""


def case_analysis_node(state: AgentState) -> dict[str, Any]:
    """LangGraph node: extract facts and create investigation plan."""
    t0 = time.time()
    logger.info("Case Analysis Agent: starting")

    claim = state["claim_case"]

    # Build the user message with the claim data
    user_msg = f"""Analyze this health insurance claim case and create an investigation plan.

CLAIM CASE:
{_format_claim(claim)}

Generate the investigation plan as JSON."""

    client = get_fast_client()
    result = client.generate_json(
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ]
    )

    duration_ms = int((time.time() - t0) * 1000)
    logger.info(
        "Case Analysis Agent: identified %d dimensions in %dms",
        len(result.get("decision_dimensions", [])),
        duration_ms,
    )

    trace_entry = {
        "agent_name": "Case Analysis Agent",
        "action": f"Extracted facts, identified {len(result.get('decision_dimensions', []))} dimensions",
        "duration_ms": duration_ms,
        "retrieval_count": 0,
        "status": "completed",
        "details": {
            "dimensions": result.get("decision_dimensions", []),
            "missing_fields_count": len(result.get("missing_fields", [])),
        },
    }

    return {
        "case_facts": result.get("case_facts", {}),
        "decision_dimensions": result.get("decision_dimensions", []),
        "investigation_plan": result.get("investigation_plan", []),
        "missing_fields": result.get("missing_fields", []),
        "trace": state.get("trace", []) + [trace_entry],
    }


def _format_claim(claim: dict) -> str:
    """Format claim dict for the LLM prompt."""
    import json

    return json.dumps(claim, indent=2, default=str)
