"""Unit tests for citation resolver and canonical policy grounding verification."""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import pytest
from retrieval.citation_resolver import resolve_citations, verify_limits_references


def test_resolve_citations_success():
    """Verify valid citations resolve to canonical text and page."""
    mock_chunks = {
        "chunk_p12_c1": {
            "chunk_id": "chunk_p12_c1",
            "page": 12,
            "section": "Section 4.1 Cataract Treatment",
            "text": "The company shall pay for Cataract surgery up to a limit of Rs. 40,000 per eye.",
        }
    }

    citations = [
        {
            "claim": "Cataract surgery limit applies",
            "chunk_id": "chunk_p12_c1",
        }
    ]

    resolved, errors = resolve_citations(citations, chunks_by_id=mock_chunks)

    assert len(errors) == 0
    assert len(resolved) == 1
    assert resolved[0]["page"] == 12
    assert resolved[0]["chunk_text"] == mock_chunks["chunk_p12_c1"]["text"]
    assert "Section 4.1" in resolved[0]["section"]


def test_resolve_citations_missing_chunk():
    """Verify citation referencing nonexistent chunk returns error."""
    citations = [
        {
            "claim": "Dental treatment exclusion",
            "chunk_id": "nonexistent_chunk_id",
        }
    ]

    resolved, errors = resolve_citations(citations, chunks_by_id={})
    assert len(errors) == 1
    assert "does not exist in indexed policy" in errors[0]


def test_resolve_citations_unsupported_claim():
    """Verify warning when chunk text has zero keyword overlap with claim."""
    mock_chunks = {
        "chunk_p5_c2": {
            "chunk_id": "chunk_p5_c2",
            "page": 5,
            "section": "General Definitions",
            "text": "Hospital means any institution established for in-patient care.",
        }
    }

    citations = [
        {
            "claim": "Maternity expenses are covered after twenty four months",
            "chunk_id": "chunk_p5_c2",
        }
    ]

    resolved, errors = resolve_citations(citations, chunks_by_id=mock_chunks)
    assert len(errors) == 1
    assert "does not support claim" in errors[0]
