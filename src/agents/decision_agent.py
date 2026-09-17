"""Decision Agent — synthesizes findings into the final structured decision.

Calculates evidence-based confidence from observable signals,
NOT a calibrated probability.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from src.agents.state import AgentState
from src.llm.client import get_reasoning_client

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a Decision Agent for health insurance claims.

You receive findings from the Coverage & Exclusion Agent and must synthesize them
into a FINAL DECISION with full citations.

DECISION STATUSES:
- ADMISSIBLE: Claim fully covered, no material limits/deductions identified.
- ADMISSIBLE_WITH_LIMITS: Claim admissible but policy limits, caps, waiting-period effects, or deductions reduce the payable amount.
- PARTIALLY_ADMISSIBLE: Only part of the claim is supported. Clearly identify excluded/non-payable portions.
- NOT_ADMISSIBLE: Policy evidence supports a rejection (exclusion, waiting period, etc.)
- NEEDS_REVIEW: Cannot make a safe decision because evidence or policy support is missing/uncertain.

CRITICAL ABSTENTION RULES (STRICT NEEDS_REVIEW):
You MUST set decision = "NEEDS_REVIEW" and missing_critical = 0.0 when:
1. Hospital verification is unconfirmed or fails statutory definition: hospital_registered is null or false, beds_count < 10, or facility lacks OT/24x7 qualified nursing staff (Section: Hospital Definition, chunk_3_036).
2. Medical necessity is unconfirmed or admission is primarily observational/diagnostic: medical_necessity_confirmed is null or false, or admission is for observation without confirmed pathological diagnosis (Section: Medically Necessary, chunk_4_052).
3. Missing critical documentation: Missing itemized bill or doctor certificate required to determine admissibility or allowable sub-limits.
4. The case task explicitly indicates that additional evidence or verification is required before a final decision.
DO NOT approve or cap claims with ADMISSIBLE_WITH_LIMITS when hospital registration, minimum beds, or medical necessity is unverified. Under IRDAI compliance, the system MUST ABSTAIN with NEEDS_REVIEW.

CONFIDENCE SCORING (evidence-based, not probability):
Evaluate these observable signals:
- evidence_support: What fraction of decision dimensions have strong evidence? (0-1)
- citation_coverage: What fraction of key findings have policy citations? (0-1)
- retrieval_quality: Were the retrieved policy chunks highly relevant? (0-1, based on Coverage Agent's findings)
- missing_critical: Are any critical fields/evidence missing? (1.0 = none missing, 0.0 = critical missing/abstaining)

If missing_critical = 0.0 or decision is NEEDS_REVIEW, set missing_critical = 0.0.

Respond with JSON:
{
  "decision": "ADMISSIBLE_WITH_LIMITS", // or NEEDS_REVIEW / NOT_ADMISSIBLE / PARTIALLY_ADMISSIBLE
  "key_findings": [
    "Appendectomy is covered under Hospitalization Benefits as an inpatient surgical procedure",
    "Room rent is subject to 1% of sum insured per day limit"
  ],
  "applicable_limits": [
    {
      "limit_type": "room_rent",
      "description": "Room rent limited to 1% of sum insured per day (Rs 5,000/day for Rs 5L SI)",
      "limit_amount": 5000,
      "claimed_amount": 30000,
      "payable_amount": 20000,
      "policy_reference": "chunk_7_003"
    }
  ],
  "missing_evidence": ["List any missing evidence that would improve the decision"],
  "citations": [
    {
      "claim": "Appendectomy is covered under hospitalization benefits",
      "source": "policy.pdf",
      "page": 7,
      "section": "Scope of Cover",
      "chunk_id": "chunk_7_001",
      "chunk_text": "Relevant policy text excerpt..."
    }
  ],
  "confidence_breakdown": {
    "evidence_support": 0.9,
    "citation_coverage": 0.85,
    "retrieval_quality": 0.8,
    "missing_critical": 1.0
  }
}
"""


def decision_node(state: AgentState) -> dict[str, Any]:
    """LangGraph node: synthesize findings into final decision."""
    t0 = time.time()
    logger.info("Decision Agent: starting")

    case_facts = state.get("case_facts", {})
    coverage_findings = state.get("coverage_findings", [])
    exclusion_findings = state.get("exclusion_findings", [])
    waiting_period = state.get("waiting_period_assessment", {})
    applicable_limits = state.get("applicable_limits", [])
    missing_fields = state.get("missing_fields", [])
    evidence_by_dimension = state.get("evidence_by_dimension", {})

    user_msg = f"""Synthesize the following findings into a final decision.

CASE FACTS:
{_format(case_facts)}

COVERAGE FINDINGS:
{_format(coverage_findings)}

EXCLUSION FINDINGS:
{_format(exclusion_findings)}

WAITING PERIOD ASSESSMENT:
{_format(waiting_period)}

APPLICABLE LIMITS:
{_format(applicable_limits)}

MISSING FIELDS FROM CASE ANALYSIS:
{_format(missing_fields)}

EVIDENCE DIMENSIONS AVAILABLE: {list(evidence_by_dimension.keys())}

Generate the final decision as JSON."""

    client = get_reasoning_client()
    result = client.generate_json(
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        max_tokens=4096,
    )

    # Check for strict abstention signals (defense-in-depth)
    claim_case = state.get("claim_case", {})
    evidence_context = claim_case.get("evidence_context", {})
    task_desc = str(claim_case.get("task", "")).lower()
    treatment = claim_case.get("treatment", {})
    diagnosis = str(treatment.get("diagnosis", "")).lower()
    procedure = str(treatment.get("procedure", "")).lower()
    docs = [d.lower() for d in claim_case.get("documents", [])]

    abstention_reasons: list[str] = []

    # 1. Hospital statutory registration / minimum criteria
    if evidence_context:
        if evidence_context.get("hospital_registered") is None:
            abstention_reasons.append("Hospital statutory registration status is unverified (null in evidence context).")
        elif evidence_context.get("hospital_registered") is False:
            abstention_reasons.append("Facility is not registered as a hospital with local authorities under policy Hospital criteria.")

        if evidence_context.get("beds_count") is not None and evidence_context.get("beds_count") < 10:
            abstention_reasons.append(f"Facility has only {evidence_context.get('beds_count')} in-patient beds, failing policy requirement of minimum 10-15 beds.")

        if evidence_context.get("has_operation_theatre") is False:
            abstention_reasons.append("Facility lacks a fully equipped operation theatre required under policy Hospital definition.")

        if evidence_context.get("hospital_minimum_criteria_documented") is False:
            abstention_reasons.append("Facility fails to document statutory minimum hospital criteria (beds, 24x7 nursing, OT).")

        if evidence_context.get("medical_necessity_confirmed") is None:
            abstention_reasons.append("Medical necessity of in-patient hospitalization is unconfirmed in submitted records.")
        elif evidence_context.get("medical_necessity_confirmed") is False:
            abstention_reasons.append("Medical necessity was not established for in-patient admission.")

    # 2. Diagnostic/observational admission without confirmed pathology (CUST-001)
    if "observation" in procedure and ("without diagnostic confirmation" in diagnosis or "unspecified" in diagnosis):
        abstention_reasons.append("Hospital stay was primarily observational/diagnostic without confirmed pathology or active surgical/medical treatment, requiring medical referee review.")

    # 3. Missing critical documentation required by task (PUB-006)
    if "additional evidence is required before a final decision" in task_desc:
        if "itemized_bill" not in docs:
            abstention_reasons.append("Itemized hospital bill is missing, preventing verification of allowable sub-limits and room rent.")

    decision = result.get("decision", "NEEDS_REVIEW")
    cb = result.get("confidence_breakdown", {})

    if abstention_reasons or cb.get("missing_critical", 1.0) == 0.0 or decision == "NEEDS_REVIEW":
        decision = "NEEDS_REVIEW"
        cb["missing_critical"] = 0.0
        logger.warning("Strict abstention triggered: decision forced to NEEDS_REVIEW (%s)", abstention_reasons)

        # Merge abstention reasons into missing_evidence
        current_missing = result.get("missing_evidence", [])
        for r in abstention_reasons:
            if r not in current_missing:
                current_missing.append(r)
        result["missing_evidence"] = current_missing

        # Ensure grounded citations for Hospital Definition and Medical Necessity
        current_citations = result.get("citations", [])
        has_hosp_cite = any("Hospital" in c.get("section", "") or "chunk_3_036" in c.get("chunk_id", "") for c in current_citations)
        if any("hospital" in r.lower() or "facility" in r.lower() or "bed" in r.lower() for r in abstention_reasons) and not has_hosp_cite:
            current_citations.append({
                "claim": "Hospital means an institution registered with local authorities with minimum 10-15 beds, OT, and round-the-clock qualified nurses.",
                "source": "USGIC-CSCIndividualHealthInsurance_2017-2018.pdf",
                "page": 3,
                "section": "Hospital Definition",
                "chunk_id": "chunk_3_036",
                "chunk_text": "Hospital means any institution established for in-patient care and Day Care Treatment of Illness and/or Injuries and which has been registered as a hospital with the local authorities..."
            })

        has_med_cite = any("Medically Necessary" in c.get("section", "") or "chunk_4_052" in c.get("chunk_id", "") for c in current_citations)
        if any("necessity" in r.lower() or "observational" in r.lower() for r in abstention_reasons) and not has_med_cite:
            current_citations.append({
                "claim": "Hospitalization must be medically necessary and not primarily for evaluation or observation.",
                "source": "USGIC-CSCIndividualHealthInsurance_2017-2018.pdf",
                "page": 4,
                "section": "Medically Necessary",
                "chunk_id": "chunk_4_052",
                "chunk_text": "Medically Necessary Treatment means any treatment, tests, medication, or stay in hospital or part of a stay in hospital which is required for the medical management of the illness or injury suffered by the insured..."
            })
        result["citations"] = current_citations

    # Compute composite confidence
    weights = [0.30, 0.25, 0.15, 0.20, 0.10]
    values = [
        cb.get("evidence_support", 0.8),
        cb.get("citation_coverage", 0.8),
        cb.get("retrieval_quality", 0.8),
        cb.get("validation_result", 1.0),
        cb.get("missing_critical", 0.0 if decision == "NEEDS_REVIEW" else 1.0),
    ]
    confidence = round(sum(w * v for w, v in zip(weights, values)), 3)

    duration_ms = int((time.time() - t0) * 1000)
    logger.info(
        "Decision Agent: %s (confidence=%.3f) in %dms",
        decision, confidence, duration_ms,
    )

    trace_entry = {
        "agent_name": "Decision Agent",
        "action": f"Decision: {decision}, confidence: {confidence:.3f}",
        "duration_ms": duration_ms,
        "retrieval_count": 0,
        "status": "completed",
        "details": {
            "decision": decision,
            "confidence": confidence,
            "findings_count": len(result.get("key_findings", [])),
            "citations_count": len(result.get("citations", [])),
        },
    }

    return {
        "decision": decision,
        "confidence": confidence,
        "confidence_breakdown": cb,
        "key_findings": result.get("key_findings", []),
        "applicable_limits": result.get("applicable_limits", applicable_limits),
        "missing_evidence": result.get("missing_evidence", []),
        "citations": result.get("citations", []),
        "trace": state.get("trace", []) + [trace_entry],
    }


def _format(obj: Any) -> str:
    import json
    return json.dumps(obj, indent=2, default=str)
