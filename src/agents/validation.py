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

from src.retrieval.citation_resolver import (
    get_indexed_chunks,
    resolve_citations,
    verify_abstention_evidence,
    verify_limits_references,
)

SYSTEM_PROMPT = """You are a strict Validation Agent for a health insurance claim decision system.

Your job is to independently verify whether every KEY FINDING and DECISION CLAIM
is ACTUALLY SUPPORTED by the cited policy evidence.

Inputs you will receive:
1. Final Decision and Key Findings
2. Applicable Limits and Deductions
3. Citations provided (with CANONICAL indexed chunk text and chunk_id)
4. Case facts

VERIFICATION RULES:
1. Every material finding must be verified against the cited canonical chunk_text or available evidence.
2. If a finding claims a specific limit (e.g., "1% room rent limit", "30-day waiting period", "cosmetic surgery excluded"), the cited chunk MUST explicitly state or clearly imply this.
3. If an LLM hallucinated a policy clause or made a claim contradictory to the cited policy text, flag it as UNSUPPORTED.
4. If a claim has no citation or references a nonexistent chunk, flag it as UNSUPPORTED.
5. If the decision is NEEDS_REVIEW (abstention): verify that the stated reasons for abstention (e.g. unverified hospital registration, lack of minimum beds, unconfirmed medical necessity, missing itemized bills) are genuine and justified under policy clauses.

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
    """LangGraph node: validate decision findings against canonical indexed evidence."""
    t0 = time.time()
    retry_count = state.get("retry_count", 0)
    logger.info("Validation Agent: validating decision (attempt %d)", retry_count + 1)

    decision = state.get("decision", "")
    key_findings = state.get("key_findings", [])
    applicable_limits = state.get("applicable_limits", [])
    raw_citations = state.get("citations", [])
    case_facts = state.get("case_facts", {})
    claim_case = state.get("claim_case", case_facts)
    missing_evidence = state.get("missing_evidence", [])

    # 1. CANONICAL CITATION RESOLUTION: Resolve chunk_ids against actual indexed chunks
    chunks_by_id = get_indexed_chunks()
    resolved_citations, citation_errors = resolve_citations(raw_citations, chunks_by_id)
    limit_errors = verify_limits_references(applicable_limits, chunks_by_id)

    # 2. ABSTENTION FACTUALITY CHECK: For NEEDS_REVIEW, verify that evidentiary gaps are real
    abstention_errors = []
    if decision == "NEEDS_REVIEW":
        is_justified, notes = verify_abstention_evidence(claim_case, missing_evidence)
        if not is_justified:
            abstention_errors.append(
                "NEEDS_REVIEW abstention is unjustified: case facts do not demonstrate missing documents or failed criteria."
            )

    code_level_errors = citation_errors + limit_errors + abstention_errors

    # 3. LLM GROUNDING EVALUATION: Send actual canonical chunk text to LLM
    user_msg = f"""Please validate the following claim adjudication result against CANONICAL indexed policy evidence:

DECISION: {decision}

KEY FINDINGS:
{_format(key_findings)}

APPLICABLE LIMITS:
{_format(applicable_limits)}

CANONICAL CITATIONS (from policy index):
{_format(resolved_citations)}

CASE FACTS & EVIDENCE CONTEXT:
{_format(claim_case)}

MISSING EVIDENCE IDENTIFIED:
{_format(missing_evidence)}

CODE RESOLVER CHECKS:
{_format(code_level_errors)}

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
    llm_unsupported = result.get("unsupported_claims", [])
    feedback = result.get("feedback", [])

    # Merge code-level citation resolution errors with LLM findings
    all_unsupported = list(set(code_level_errors + llm_unsupported))

    if all_unsupported:
        status = "FAIL"
    else:
        status = "PASS"

    duration_ms = int((time.time() - t0) * 1000)
    logger.info(
        "Validation Agent: %s (unsupported=%d, code_errors=%d) in %dms",
        status, len(all_unsupported), len(code_level_errors), duration_ms
    )

    trace_entry = {
        "agent_name": "Validation Agent",
        "action": f"Validation verdict: {status} ({len(all_unsupported)} unsupported claims)",
        "duration_ms": duration_ms,
        "retrieval_count": len(resolved_citations),
        "status": status.lower(),
        "details": {
            "status": status,
            "unsupported_count": len(all_unsupported),
            "code_errors_count": len(code_level_errors),
            "retry_count": retry_count,
            "reasoning": result.get("reasoning_summary", ""),
        },
    }

    validation_dict = {
        "status": status,
        "unsupported_claims": all_unsupported,
        "feedback": feedback,
    }

    cb = state.get("confidence_breakdown", {})
    if cb:
        cb["validation_result"] = 1.0 if status == "PASS" else (0.5 if retry_count > 0 else 0.0)

    return {
        "validation": validation_dict,
        "validation_feedback": feedback,
        "citations": resolved_citations,  # return canonical resolved citations
        "retry_count": retry_count + 1 if status == "FAIL" else retry_count,
        "trace": state.get("trace", []) + [trace_entry],
    }


def _format(obj: Any) -> str:
    import json
    return json.dumps(obj, indent=2, default=str)
