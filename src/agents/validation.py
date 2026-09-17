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

SYSTEM_PROMPT = """You are a strict but fair Validation Agent for a health insurance claim decision system.

Your job is to independently verify whether KEY FINDINGS, DECISION CLAIMS, and LIMITS
are FACTUALLY GROUNDED in the cited canonical policy evidence.

Inputs you will receive:
1. Final Decision and Key Findings
2. Applicable Limits and Deductions
3. Citations provided (with CANONICAL indexed chunk text and chunk_id)
4. Case facts and missing evidence

EVIDENCE INTERPRETATION STANDARDS:
1. GENERAL COVERAGE SCOPE:
   Under health insurance policies, the Scope of Cover (chunk_7_099) covers hospitalization expenses "when You sustain any injury or contact any disease". Health policies cover diseases generally and DO NOT enumerate thousands of individual medical diagnoses. Therefore:
   - Stating that inpatient treatment for illnesses (e.g., Cancer, Acute Gastroenteritis, Septoplasty, Appendicitis, Pneumonia) is covered under chunk_7_099 and medically necessary under chunk_4_052 is FULLY SUPPORTED, provided no policy exclusion applies.
   - DO NOT flag a finding as unsupported merely because the policy clause does not explicitly name the patient's specific diagnosis.

2. POLICY EXCLUSIONS:
   - Clause 5 in chunk_9_115 explicitly excludes "cosmetic or aesthetic treatment of any description (including any complications arising thereof), plastic surgery except those relating to treatment of Injury or Disease". Citing chunk_9_115 for cosmetic surgery or aesthetic treatment exclusion is FULLY SUPPORTED. Do NOT flag this as unsupported because clause 5 also mentions circumcision or vaccinations.

3. WAITING PERIODS:
   - Under chunk_9_115, a 30-day initial waiting period applies unless continuously insured, and a 1-year waiting period applies to specified conditions (cataract, hernia, piles, sinusitis, joint replacement, cysts/tumors unless malignant).
   - If tenure exceeds these periods, or if cancer is malignant ("unless malignant" under item xi), or if the condition is not pre-existing, stating that waiting periods are satisfied is SUPPORTED.

4. DOMICILIARY TREATMENT:
   - Domiciliary treatment (chunk_2_028) takes place at home when hospital rooms are unavailable or patient cannot be moved, and is subject to the 20% sub-limit (chunk_7_102). Domiciliary treatment by definition occurs at home rather than in a 24-hour hospital facility.

5. EXPERIMENTAL / UNPROVEN TREATMENT:
   - When treatment is experimental or investigational (treatment.experimental == true or procedure is experimental therapy), a rejection of NOT_ADMISSIBLE citing chunk_6_077 (unproven/experimental treatment definition), chunk_4_052 (medically necessary standards), or chunk_10_117 (unapproved treatments) is FULLY SUPPORTED and must be marked PASS.

6. HOSPITAL QUALIFICATION & ADMINISTRATIVE REVIEW (NEEDS_REVIEW):
   - A missing or incomplete hospital-qualification record requires NEEDS_REVIEW; do not infer final non-compliance unless the case expressly confirms each failed policy criterion.
   - When hospital_registered is null/false or hospital criteria (10-15 beds, OT, nursing under chunk_3_036) are unverified or undocumented, adjudicating as NEEDS_REVIEW is the EXPECTED, CORRECT, and SAFE procedure. Do NOT flag NEEDS_REVIEW as an unjustified abstention, and do NOT claim that lack of proof warrants immediate outright denial.

7. PROCEDURAL FINDINGS IN NEEDS_REVIEW:
   - In a NEEDS_REVIEW decision, findings that explain what evidence is missing (e.g. itemized bills, physician admission rationale, discharge summary) or state that coverage and deductions cannot be finalized until verified are valid procedural statements. Do NOT flag them as unsupported policy claims.

8. GENUINE FAILURES TO FLAG (status = "FAIL"):
   - A cited chunk_id does not exist in the index.
   - The cited chunk text contradicts the claim (e.g., claiming 100% room rent when chunk states 1%).
   - An LLM invents a non-existent exclusion or policy rule.
   - An abstention (NEEDS_REVIEW) is claimed without any genuine evidentiary or document defect (Note: missing itemized bill, unverified medical necessity, unverified hospital registration, or missing clinical records ARE genuine defects justifying NEEDS_REVIEW).

If all material statements are substantively grounded in policy provisions or procedural evidence rules: status = "PASS", unsupported_claims = [].
Only if there are genuine contradictions, invented limits, or ungrounded rules: status = "FAIL".

Output format (strict JSON):
{
  "status": "PASS", // or "FAIL"
  "unsupported_claims": [],
  "feedback": [],
  "reasoning_summary": "1-2 sentence assessment of policy evidence fidelity."
}
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
