"""Canonical Citation Resolver & Grounding Verification.

Resolves LLM citations against actual indexed policy chunks in chunks.json.
Guarantees provenance, canonical text injection, and factual verification.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_CHUNKS_CACHE: Optional[Dict[str, Dict[str, Any]]] = None


def get_indexed_chunks(chunks_path: str = "data/index/chunks.json") -> Dict[str, Dict[str, Any]]:
    """Load and cache all indexed policy chunks keyed by chunk_id."""
    global _CHUNKS_CACHE
    if _CHUNKS_CACHE is not None:
        return _CHUNKS_CACHE

    p = Path(chunks_path)
    if not p.exists():
        logger.error("Chunks index file not found at %s", chunks_path)
        return {}

    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)

    _CHUNKS_CACHE = {c["chunk_id"]: c for c in data if "chunk_id" in c}
    logger.info("Loaded %d policy chunks into canonical citation resolver.", len(_CHUNKS_CACHE))
    return _CHUNKS_CACHE


def resolve_citations(
    citations: List[Dict[str, Any]],
    chunks_by_id: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Validate and resolve citations against actual indexed policy chunks.

    Returns:
        (resolved_citations, unsupported_errors)
    """
    if chunks_by_id is None:
        chunks_by_id = get_indexed_chunks()

    resolved: List[Dict[str, Any]] = []
    errors: List[str] = []

    for idx, cit in enumerate(citations):
        chunk_id = cit.get("chunk_id", "").strip()
        claim = cit.get("claim", "").strip()

        if not chunk_id:
            errors.append(f"Citation #{idx+1} for claim '{claim[:60]}' has no chunk_id.")
            continue

        if chunk_id not in chunks_by_id:
            errors.append(
                f"Citation #{idx+1} references invalid chunk_id '{chunk_id}' that does not exist in indexed policy."
            )
            continue

        actual_chunk = chunks_by_id[chunk_id]
        canonical_text = actual_chunk.get("text", "")
        canonical_page = actual_chunk.get("page", cit.get("page", 1))
        canonical_section = actual_chunk.get("section") or actual_chunk.get("heading_path", "")

        # Invalidate if text is unexpectedly empty
        if not canonical_text.strip():
            errors.append(f"Chunk '{chunk_id}' contains empty text in the policy index.")
            continue

        # Check basic lexical or concept support in canonical text
        # Extract meaningful words (len > 3) from claim
        claim_keywords = [w.lower() for w in claim.replace(",", " ").replace(".", " ").split() if len(w) > 3]
        text_lower = canonical_text.lower()
        keyword_hits = sum(1 for kw in claim_keywords if kw in text_lower)
        hit_ratio = keyword_hits / max(len(claim_keywords), 1)

        # Flag if claim is completely absent from the actual chunk text
        if claim_keywords and hit_ratio < 0.15:
            errors.append(
                f"Chunk '{chunk_id}' (Page {canonical_page}) text does not support claim: '{claim[:80]}' (keyword overlap {hit_ratio:.0%})."
            )

        resolved_cit = {
            "claim": claim,
            "source": "USGIC-CSCIndividualHealthInsurance_2017-2018.pdf",
            "page": canonical_page,
            "section": canonical_section,
            "chunk_id": chunk_id,
            "chunk_text": canonical_text,
        }
        resolved.append(resolved_cit)

    return resolved, errors


def verify_limits_references(
    applicable_limits: List[Dict[str, Any]],
    chunks_by_id: Optional[Dict[str, Dict[str, Any]]] = None,
) -> List[str]:
    """Verify that every applicable limit cites a valid policy chunk that discusses limits."""
    if chunks_by_id is None:
        chunks_by_id = get_indexed_chunks()

    errors: List[str] = []
    for lim in applicable_limits:
        ref = lim.get("policy_reference", "").strip()
        desc = lim.get("description", "")

        if not ref:
            errors.append(f"Applicable limit '{desc[:60]}' has no policy_reference chunk_id.")
            continue

        if ref not in chunks_by_id:
            errors.append(f"Applicable limit '{desc[:60]}' references invalid chunk_id '{ref}'.")
            continue

        chunk_text = chunks_by_id[ref].get("text", "").lower()
        # Verify that the chunk actually mentions limits, room rent, expenses, charges, or sum insured
        limit_terms = ["limit", "room", "rent", "sum insured", "sub-limit", "exceed", "charge", "percentage", "%", "admissible", "payable"]
        if not any(term in chunk_text for term in limit_terms):
            errors.append(f"Chunk '{ref}' cited for limit '{desc[:50]}' does not mention limits or eligible charges.")

    return errors


def verify_abstention_evidence(
    claim_case: Dict[str, Any],
    missing_evidence: List[str],
) -> Tuple[bool, List[str]]:
    """Verify that an abstention (NEEDS_REVIEW) is factually supported by case facts."""
    evidence_context = claim_case.get("evidence_context", {})
    documents = [d.lower() for d in claim_case.get("documents", [])]
    treatment = claim_case.get("treatment", {})
    task_desc = str(claim_case.get("task", "")).lower()

    justifications: List[str] = []
    falsifications: List[str] = []

    # Check for genuine evidentiary gaps
    if evidence_context.get("hospital_registered") is None:
        justifications.append("hospital_registered is unverified (null)")
    elif evidence_context.get("hospital_registered") is False:
        justifications.append("hospital is explicitly not registered under policy criteria")

    if evidence_context.get("beds_count") is not None and evidence_context.get("beds_count") < 10:
        justifications.append(f"hospital has only {evidence_context.get('beds_count')} beds (<10 minimum)")

    if evidence_context.get("has_operation_theatre") is False:
        justifications.append("hospital lacks operation theatre")

    if evidence_context.get("hospital_minimum_criteria_documented") is False:
        justifications.append("facility fails to document statutory criteria")

    if evidence_context.get("medical_necessity_confirmed") is None:
        justifications.append("medical necessity is unconfirmed (null)")
    elif evidence_context.get("medical_necessity_confirmed") is False:
        justifications.append("medical necessity was not established")

    if "additional evidence is required before a final decision" in task_desc:
        if "itemized_bill" not in documents:
            justifications.append("itemized_bill missing when required by task")

    proc = str(treatment.get("procedure", "")).lower()
    diag = str(treatment.get("diagnosis", "")).lower()
    if "observation" in proc and ("without diagnostic confirmation" in diag or "unspecified" in diag):
        justifications.append("admission is observational without confirmed pathological diagnosis")

    if not justifications:
        falsifications.append(
            "NEEDS_REVIEW abstention lacks factual evidentiary justification in case facts or evidence_context."
        )

    return len(justifications) > 0, justifications + falsifications

