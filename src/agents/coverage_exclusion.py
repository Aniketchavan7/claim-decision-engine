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

CANONICAL POLICY CLAUSE REFERENCE (Use these exact chunk IDs):
- In-patient 24-hour admission requirement: chunk_3_035 ("admission in a Hospital for a minimum period of 24 consecutive hours")
- Day Care Treatment waiver of 24 hours (140 procedures including cataract/eye surgery, dialysis, chemo): chunk_7_104 and chunk_2_026
- Hospital Definition (registered with local authorities, min 10-15 beds, OT, 24/7 nursing): chunk_3_036
- Medically Necessary Treatment: chunk_4_052
- Scope of Cover (Room, Boarding, Nursing expenses covered): chunk_7_099
- Sub-limits (Room rent 1% SI/day, ICU 2% SI/day, Doctor/Surgeon fees 25% SI, Surgical/medicines/OT 40% SI): chunk_7_100
- Ambulance Charges (1.0% of Basic Sum Insured or Rs 1,000/- whichever is less): chunk_8_111
- Domiciliary Hospitalization (20% SI sub-limit, >3 days, room unavailable or patient immobile): chunk_7_102 and chunk_2_028
- Agreed Package Charges (75% SI cap): chunk_7_103. CRITICAL: ONLY apply this cap if the claim was explicitly billed under agreed package charges. If bills are itemized, package charge limit does NOT apply.
- Pre/Post Hospitalization (30 days pre, 60 days post) and Sum Insured limits: chunk_8_112
- Initial 30-day Waiting Period & 1-Year Specific Disease Waiting Period (Clause 3: cataract, hernia, piles, sinusitis, joint replacement, myomectomy, etc.): chunk_9_115
- Pre-Existing Diseases (48-month waiting period): chunk_8_108; Portability credit: chunk_8_109; PED definition: chunk_5_062
- General Exclusions (Cosmetic/aesthetic treatment of any description, plastic surgery except for injury/disease, dental, circumcision, spectacles, HIV/AIDS, pregnancy): chunk_9_115 (item 5 explicitly excludes cosmetic/plastic surgery)
- Unproven / Experimental Treatment Definition & Exclusion: chunk_6_077 ("Unproven/Experimental Treatment means a treatment, including drug Experimental therapy, which is not based on established medical practice in India"), chunk_4_052 (Medically Necessary Treatment: must conform to accepted professional medical standards), and chunk_10_117 (item 14: treatments not approved by Medical Council).

CRITICAL CITATION RULES:
1. Base every finding on the ACTUAL policy text provided. Quote or reference it.
2. WAITING PERIODS:
   - If the condition is NOT pre-existing (pre_existing: false), state: "Condition is not pre-existing (chunk_5_062), so the 48-month PED waiting period (chunk_8_108) is not applicable. The initial 30-day waiting period and 1-year specific waiting period are satisfied under chunk_9_115."
   - NEVER state that <48 months of coverage satisfies the 48-month PED waiting period of chunk_8_108.
3. EXCLUSIONS:
   - General Exclusions (cosmetic surgery, aesthetic treatment, plastic surgery) are in chunk_9_115 item 5. Do NOT cite chunk_8_107 (header only) or chunk_2_024 (dental only) for cosmetic exclusions.
4. EXPERIMENTAL / UNPROVEN TREATMENT:
   - When treatment.experimental is true or procedure is experimental therapy, the treatment is excluded from coverage.
   - Cite chunk_6_077 (unproven/experimental treatment definition), chunk_4_052 (failure to meet medically necessary professional standards), and chunk_10_117.
   - Formulate an exclusion finding (category: "exclusion", supported: true) indicating the claim is NOT_ADMISSIBLE.
5. HOSPITALIZATION & 24-HOUR REQUIREMENT:
   - For standard 24-hour minimum stay: cite chunk_3_035.
   - For Day Care waiver: cite chunk_7_104.
6. DOMICILIARY HOSPITALIZATION:
   - Domiciliary treatment condition under chunk_2_028 is satisfied if EITHER patient_cannot_be_moved is true OR hospital_room_unavailable is true.
   - When hospital_room_unavailable is true, the policy condition is met. Reason about coverage and apply the 20% Basic Sum Insured sub-limit (chunk_7_102). Do not claim evidence is missing for room unavailability.
7. If evidence is insufficient to make a determination, say "INSUFFICIENT_EVIDENCE" for that dimension.
8. Each finding must reference the chunk_id(s) that support it.

Respond with JSON:
{
  "coverage_findings": [
    {
      "description": "...",
      "category": "coverage",
      "dimension": "coverage",
      "supported": true,
      "evidence_references": ["chunk_3_035", "chunk_7_099"]
    }
  ],
  "exclusion_findings": [
    {
      "description": "Cosmetic surgery is explicitly excluded under General Exclusions item 5",
      "category": "exclusion",
      "dimension": "exclusion",
      "supported": true,
      "evidence_references": ["chunk_9_115"]
    }
  ],
  "waiting_period_assessment": {
    "initial_waiting_period_applies": false,
    "specific_disease_waiting_applies": false,
    "ped_waiting_applies": false,
    "waiting_period_satisfied": true,
    "details": "...",
    "evidence_references": ["chunk_9_115"]
  },
  "applicable_limits": [
    {
      "limit_type": "room_rent",
      "description": "Room rent limited to 1% of sum insured per day",
      "limit_amount": 5000,
      "claimed_amount": 30000,
      "payable_amount": 20000,
      "policy_reference": "chunk_7_100"
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
