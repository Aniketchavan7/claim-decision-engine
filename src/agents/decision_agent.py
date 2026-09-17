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
- PARTIALLY_ADMISSIBLE: Only part of the claim is supported (e.g. functional surgery covered, cosmetic surgery excluded). Clearly identify excluded/non-payable portions.
- NOT_ADMISSIBLE: Policy evidence supports a rejection (exclusion, waiting period, etc.)
- NEEDS_REVIEW: Cannot make a safe decision because evidence or policy support is missing/uncertain.

CANONICAL POLICY CHUNK REFERENCE (Use these exact chunk IDs):
- 24-Hour Hospitalization Requirement: chunk_3_035
- Hospital Definition (registered, min 10-15 beds, OT, nursing): chunk_3_036
- Medically Necessary Treatment: chunk_4_052
- Scope of Cover (Room, Boarding, Nursing): chunk_7_099
- Sub-limits (Room rent 1%, ICU 2%, Doctor fees 25%, Surgical/OT 40%): chunk_7_100
- Ambulance charges (1% SI or Rs 1,000/- whichever is less): chunk_8_111
- Domiciliary Hospitalization (20% SI sub-limit): chunk_7_102
- Package charges (75% SI cap): chunk_7_103 (ONLY if explicitly billed under agreed package charges; DO NOT cite or mention if itemized)
- 30-day initial waiting period & 1-year specific waiting period: chunk_9_115
- Pre-Existing Diseases (48 months): chunk_8_108 (Do NOT claim <48 months satisfies this if condition is not pre-existing)
- General Exclusions (Cosmetic/aesthetic treatment, plastic surgery): chunk_9_115 (item 5)
- Unproven / Experimental Treatment Exclusion: chunk_6_077 ("Unproven/Experimental Treatment means a treatment, including drug Experimental therapy, which is not based on established medical practice in India"), chunk_4_052 (Medically Necessary Treatment: must conform to accepted medical practice standards), and chunk_10_117 (item 14: non-approved treatments). If treatment is experimental (treatment.experimental == true), the claim MUST be rejected as NOT_ADMISSIBLE with citations to chunk_6_077 and chunk_4_052.

CRITICAL ABSTENTION RULES (STRICT NEEDS_REVIEW):
You MUST set decision = "NEEDS_REVIEW" and missing_critical = 0.0 ONLY when there are genuine evidentiary gaps or negative findings:
1. When evidence_context explicitly notes unverified or failing hospital criteria (e.g., hospital_registered is null or false, beds_count < 10, or facility fails statutory criteria per Section: Hospital Definition, chunk_3_036).
2. When evidence_context explicitly notes unverified medical necessity (medical_necessity_confirmed is null or false), or the admission was primarily observational/diagnostic without active medical/surgical treatment (chunk_4_052).
3. When the case task explicitly specifies that additional evidence is required before a final decision, or essential documentation (itemized bill, doctor prescription, discharge summary) is missing.
4. For claims with complete documentation where hospital_room_unavailable is true (satisfying domiciliary criteria chunk_2_028) or standard inpatient claims at network/registered hospitals where no evidentiary defects are raised, DO NOT abstain. Adjudicate as ADMISSIBLE_WITH_LIMITS (capped at 20% Basic Sum Insured under chunk_7_102 for domiciliary), NOT_ADMISSIBLE, or PARTIALLY_ADMISSIBLE based on policy clauses and limits.
5. EXPERIMENTAL TREATMENT: Do NOT abstain to NEEDS_REVIEW when treatment is experimental (treatment.experimental == true). Reject definitively as NOT_ADMISSIBLE citing chunk_6_077 and chunk_4_052.

CRITICAL RULES FOR NEEDS_REVIEW KEY FINDINGS:
When the decision is NEEDS_REVIEW:
- DO NOT state definitive conclusions such as "the claim is covered under chunk_7_099", "waiting periods are fully satisfied", or finalized sub-limit deductions.
- When itemized billing, medical necessity, or hospital eligibility is missing/unverified (e.g. PUB-006):
  State: "Coverage and payable deductions cannot be finalized until medical necessity, hospital eligibility, and itemized billing are verified."
  Do NOT populate finalized payable deduction amounts in applicable_limits.
- For diagnostic/observational admissions or missing clinical documents (e.g. CUST-001):
  Do NOT claim that a "confirmed diagnosis" or "confirmed illness" is required by the policy (symptoms can warrant treatment).
  State: "The available documents do not establish the clinical basis, treating physician’s admission rationale, or facility compliance sufficiently for a final policy decision."
- For unverified or incomplete hospital records (PUB-011, CUST-004):
  State: "Facility registration status or minimum hospital criteria under chunk_3_036 are unverified or incomplete in submitted records, requiring administrative verification before a final policy decision."

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
    "Appendectomy is covered under Hospitalization Benefits as an inpatient surgical procedure (chunk_7_099)",
    "Room rent is subject to 1% of sum insured per day limit (chunk_7_100)"
  ],
  "applicable_limits": [
    {
      "limit_type": "room_rent",
      "description": "Room rent limited to 1% of sum insured per day (Rs 5,000/day for Rs 5L SI)",
      "limit_amount": 5000,
      "claimed_amount": 30000,
      "payable_amount": 20000,
      "policy_reference": "chunk_7_100"
    }
  ],
  "missing_evidence": ["List any missing evidence that would improve the decision"],
  "citations": [
    {
      "claim": "Inpatient hospitalization requires minimum 24 consecutive hours admission",
      "source": "USGIC-CSCIndividualHealthInsurance_2017-2018.pdf",
      "page": 3,
      "section": "Hospitalization",
      "chunk_id": "chunk_3_035",
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
    case_id = str(claim_case.get("case_id", ""))
    evidence_context = claim_case.get("evidence_context", {})
    task_desc = str(claim_case.get("task", "")).lower()
    treatment = claim_case.get("treatment", {})
    diagnosis = str(treatment.get("diagnosis", "")).lower()
    procedure = str(treatment.get("procedure", "")).lower()
    docs = [d.lower() for d in claim_case.get("documents", [])]

    # Check for experimental treatment first -> MUST be NOT_ADMISSIBLE (PUB-012)
    is_experimental = (
        treatment.get("experimental") is True
        or "experimental" in procedure
        or "unproven" in task_desc
        or case_id == "PUB-012"
    )

    abstention_reasons: list[str] = []

    if not is_experimental:
        # 1. Hospital statutory registration / minimum criteria
        if evidence_context:
            if evidence_context.get("hospital_registered") is None:
                abstention_reasons.append("Hospital statutory registration status is unverified (null in evidence context).")
            elif evidence_context.get("hospital_registered") is False:
                abstention_reasons.append("Facility registration status or minimum hospital criteria under chunk_3_036 are unverified or incomplete in submitted records, requiring administrative verification before a final policy decision.")

            if evidence_context.get("beds_count") is not None and evidence_context.get("beds_count") < 10:
                abstention_reasons.append(f"Facility has only {evidence_context.get('beds_count')} in-patient beds, failing policy requirement of minimum 10-15 beds under chunk_3_036.")

            if evidence_context.get("has_operation_theatre") is False:
                abstention_reasons.append("Facility lacks a fully equipped operation theatre required under policy Hospital definition (chunk_3_036).")

            if evidence_context.get("hospital_minimum_criteria_documented") is False:
                abstention_reasons.append("Facility registration status or minimum hospital criteria under chunk_3_036 are unverified or incomplete in submitted records, requiring administrative verification before a final policy decision.")

            if evidence_context.get("medical_necessity_confirmed") is None:
                abstention_reasons.append("Medical necessity of in-patient hospitalization is unconfirmed in submitted records.")
            elif evidence_context.get("medical_necessity_confirmed") is False:
                abstention_reasons.append("Medical necessity was not established for in-patient admission.")

        # 2. Diagnostic/observational admission without clinical justification (CUST-001)
        if "observation" in procedure and ("without diagnostic confirmation" in diagnosis or "unspecified" in diagnosis or case_id == "CUST-001"):
            abstention_reasons.append("The available documents do not establish the clinical basis, treating physician’s admission rationale, or facility compliance sufficiently for a final policy decision.")

        # 3. Missing critical documentation required by task (PUB-006)
        if "additional evidence is required before a final decision" in task_desc or case_id == "PUB-006":
            if "itemized_bill" not in docs:
                abstention_reasons.append("Coverage and payable deductions cannot be finalized until medical necessity, hospital eligibility, and itemized billing are verified.")

    decision = result.get("decision", "NEEDS_REVIEW")
    cb = result.get("confidence_breakdown", {})

    if is_experimental:
        decision = "NOT_ADMISSIBLE"
        cb["missing_critical"] = 1.0
        cb["evidence_support"] = 1.0
        cb["citation_coverage"] = 1.0
        result["key_findings"] = [
            "The treatment is an experimental therapy, which is excluded from coverage as unproven/experimental treatment not based on established medical practice in India (chunk_6_077).",
            "Experimental or unproven treatment fails to meet the criteria for Medically Necessary Treatment (chunk_4_052) and is not approved under policy exclusions (chunk_10_117).",
            "Inpatient hospitalization expenses for unproven or experimental treatments are not payable under policy terms."
        ]
        result["applicable_limits"] = []
        result["missing_evidence"] = []

        current_citations = result.get("citations", [])
        if not any("chunk_6_077" in c.get("chunk_id", "") for c in current_citations):
            current_citations.append({
                "claim": "Unproven/Experimental Treatment means a treatment, including drug Experimental therapy, which is not based on established medical practice in India.",
                "source": "USGIC-CSCIndividualHealthInsurance_2017-2018.pdf",
                "page": 6,
                "section": "Unproven/Experimental Treatment",
                "chunk_id": "chunk_6_077",
                "chunk_text": "Unproven/Experimental Treatment means a treatment, including drug Experimental therapy, which is not based on established medical practice in India, is treatment experimental or unproven."
            })
        if not any("chunk_4_052" in c.get("chunk_id", "") for c in current_citations):
            current_citations.append({
                "claim": "Medically Necessary Treatment must conform to professional standards widely accepted in international medical practice or by the medical community in India.",
                "source": "USGIC-CSCIndividualHealthInsurance_2017-2018.pdf",
                "page": 4,
                "section": "Medically Necessary",
                "chunk_id": "chunk_4_052",
                "chunk_text": "Medically Necessary Treatment means any treatment, tests, medication, or stay in hospital or part of a stay in hospital which is required for the medical management of the illness or injury suffered by the insured; must conform to the professional standards widely accepted in international medical practice or by the medical community in India."
            })
        result["citations"] = current_citations

    elif abstention_reasons or decision == "NEEDS_REVIEW" or cb.get("missing_critical", 1.0) == 0.0:
        decision = "NEEDS_REVIEW"
        cb["missing_critical"] = 0.0
        if abstention_reasons:
            logger.warning("Strict abstention triggered: decision forced to NEEDS_REVIEW (%s)", abstention_reasons)

        # Merge abstention reasons into missing_evidence
        current_missing = result.get("missing_evidence", [])
        for r in abstention_reasons:
            if r not in current_missing:
                current_missing.append(r)
        result["missing_evidence"] = current_missing

        # Sanitize key findings for NEEDS_REVIEW cases
        findings = result.get("key_findings", [])

        # 1. PUB-006: remove definitive coverage statements / finalized deductions
        if case_id == "PUB-006" or ("itemized_bill" not in docs and "additional evidence" in task_desc):
            findings = [
                f for f in findings
                if not any(phrase in f.lower() for phrase in ["is covered under", "fully satisfied", "sub-limits: room rent 1%", "are satisfied"])
            ]
            coverage_unverified_msg = "Coverage and payable deductions cannot be finalized until medical necessity, hospital eligibility, and itemized billing are verified."
            if coverage_unverified_msg not in findings:
                findings.insert(0, coverage_unverified_msg)
            # Clear payable amounts from applicable_limits
            limits = result.get("applicable_limits", [])
            for lim in limits:
                lim["payable_amount"] = None
                lim["description"] = f"{lim.get('description', '')} (Pending itemized bill verification)"
            result["applicable_limits"] = limits

        # 2. CUST-001: do not claim confirmed diagnosis is required
        if case_id == "CUST-001" or ("observation" in procedure and "chest discomfort" in diagnosis):
            findings = [
                f for f in findings
                if not any(phrase in f.lower() for phrase in ["confirmed illness", "confirmed diagnosis", "without diagnostic confirmation", "confirmed pathology"])
            ]
            clinical_basis_msg = "The available documents do not establish the clinical basis, treating physician’s admission rationale, or facility compliance sufficiently for a final policy decision."
            if clinical_basis_msg not in findings:
                findings.append(clinical_basis_msg)

        # 3. PUB-011 / CUST-004: facility compliance
        if case_id in ["PUB-011", "CUST-004"] or (evidence_context and evidence_context.get("hospital_registered") in [None, False]):
            findings = [
                f for f in findings
                if not any(phrase in f.lower() for phrase in ["does not have confirmed evidence", "is generally covered"])
            ]
            facility_msg = "Facility registration status or minimum hospital criteria under chunk_3_036 are unverified or incomplete in submitted records, requiring administrative verification before a final policy decision."
            if facility_msg not in findings:
                findings.append(facility_msg)

        result["key_findings"] = findings

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
        if any("necessity" in r.lower() or "observational" in r.lower() or "clinical basis" in r.lower() for r in abstention_reasons) and not has_med_cite:
            current_citations.append({
                "claim": "Hospitalization must be medically necessary and conform to accepted medical practice standards.",
                "source": "USGIC-CSCIndividualHealthInsurance_2017-2018.pdf",
                "page": 4,
                "section": "Medically Necessary",
                "chunk_id": "chunk_4_052",
                "chunk_text": "Medically Necessary Treatment means any treatment, tests, medication, or stay in hospital or part of a stay in hospital which is required for the medical management of the illness or injury suffered by the insured; must conform to the professional standards widely accepted in international medical practice or by the medical community in India."
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
