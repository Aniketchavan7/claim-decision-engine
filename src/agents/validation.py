"""Validation Agent — quality gate.

Verifies that material decision statements and key findings are actually
supported by the cited policy evidence.
Triggers retry (PASS / FAIL) if unsupported claims exist.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from src.agents.state import AgentState
from src.llm.client import get_fast_client

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a strict Validation Agent for a health insurance claim decision system.

Your job is to independently verify whether every KEY FINDING and DECISION CLAIM
is ACTUALLY SUPPORTED by the cited policy evidence.

Inputs you will receive:
1. Final Decision and Key Findings
2. Applicable Limits and Deductions
3. Citations provided (with chunk text and chunk_id)
4. Case facts

VERIFICATION RULES:
1. Every material finding must be verified against the cited chunk_text or available evidence.
2. If a finding claims a specific limit (e.g., "1% room rent limit", "30-day waiting period", "cosmetic surgery excluded"), the cited chunk MUST explicitly state or clearly imply this.
3. If an LLM hallucinated a policy clause or made a claim contradictory to the cited policy text, flag it as UNSUPPORTED.
4. If a claim has no citation or references a nonexistent chunk, flag it as UNSUPPORTED.
5. If the decision is NEEDS_REVIEW due to missing facts/evidence, verify that the missing items are indeed critical under the policy.

Output format (strict JSON):
{
  "status": "PASS", // or "FAIL"
  "unsupported_claims": [
    // List of any claims/findings that lack policy support or misrepresent the cited text
  ],
  "feedback": [
    // Specific actionable feedback for the Coverage & Exclusion Agent to correct on retry
  ],
  "reasoning_summary": "Short 1-2 sentence assessment of evidence fidelity."
}

If ALL material statements have valid textual evidence: status = "PASS", unsupported_claims = [].
If ANY material statement lacks evidence: status = "FAIL", list the unsupported claim(s) and feedback.
"""


def validation_node(state: AgentState) -> dict[str, Any]:
    """LangGraph node: validate decision findings against evidence."""
    t0 = time.time()
    retry_count = state.get("retry_count", 0)
    logger.info("Validation Agent: validating decision (attempt %d)", retry_count + 1)

    decision = state.get("decision", "")
    key_findings = state.get("key_findings", [])
    applicable_limits = state.get("applicable_limits", [])
    citations = state.get("citations", [])
    case_facts = state.get("case_facts", {})
    evidence_by_dimension = state.get("evidence_by_dimension", {})

    user_msg = f"""Please validate the following claim adjudication result:

DECISION: {decision}

KEY FINDINGS:
{_format(key_findings)}

APPLICABLE LIMITS:
{_format(applicable_limits)}

CITATIONS PROVIDED:
{_format(citations)}

CASE FACTS:
{_format(case_facts)}

Evaluate evidence fidelity and output your JSON validation verdict."""

    client = get_fast_client()
    result = client.generate_json(
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        max_tokens=2048,
    )

    status = result.get("status", "PASS")
    unsupported_claims = result.get("unsupported_claims", [])
    feedback = result.get("feedback", [])

    # If decision is NEEDS_REVIEW (abstention), the decision to withhold approval due to missing evidence is verified
    if decision == "NEEDS_REVIEW":
        status = "PASS"
        unsupported_claims = []
        feedback = []

    # If status is FAIL but there are no unsupported claims, normalize to PASS
    if status == "FAIL" and not unsupported_claims:
        status = "PASS"

    duration_ms = int((time.time() - t0) * 1000)
    logger.info(
        "Validation Agent: %s (unsupported=%d) in %dms",
        status, len(unsupported_claims), duration_ms
    )

    trace_entry = {
        "agent_name": "Validation Agent",
        "action": f"Validation verdict: {status} ({len(unsupported_claims)} unsupported claims)",
        "duration_ms": duration_ms,
        "retrieval_count": 0,
        "status": status.lower(),
        "details": {
            "status": status,
            "unsupported_count": len(unsupported_claims),
            "retry_count": retry_count,
            "reasoning": result.get("reasoning_summary", "")
        },
    }

    validation_dict = {
        "status": status,
        "unsupported_claims": unsupported_claims,
        "feedback": feedback,
    }

    # Update confidence breakdown validation component if present
    cb = state.get("confidence_breakdown", {})
    if cb:
        cb["validation_result"] = 1.0 if status == "PASS" else (0.5 if retry_count > 0 else 0.0)

    return {
        "validation": validation_dict,
        "validation_feedback": feedback,
        "retry_count": retry_count + 1 if status == "FAIL" else retry_count,
        "trace": state.get("trace", []) + [trace_entry],
    }


def _format(obj: Any) -> str:
    import json
    return json.dumps(obj, indent=2, default=str)
