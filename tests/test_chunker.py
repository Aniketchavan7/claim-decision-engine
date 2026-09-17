import pytest
from src.ingestion.pdf_parser import TextBlock
from src.ingestion.chunker import chunk_document


def test_chunker_basic():
    blocks = [
        TextBlock(text="Section 1: Scope of Cover", page=1, is_heading=True, font_size=16.0, heading_level=1),
        TextBlock(text="This section outlines the basic inpatient hospitalization benefits.", page=1, is_heading=False, font_size=10.0, heading_level=0),
        TextBlock(text="Section 2: Exclusions", page=2, is_heading=True, font_size=16.0, heading_level=1),
        TextBlock(text="Cosmetic surgeries and unproven procedures are excluded from policy scope.", page=2, is_heading=False, font_size=10.0, heading_level=0),
    ]

    chunks = chunk_document(blocks, max_tokens=200, overlap_tokens=20)
    assert len(chunks) >= 2
    assert any("Scope of Cover" in c.text or "Scope of Cover" in c.section for c in chunks)
    assert any("Exclusions" in c.text or "Exclusions" in c.section for c in chunks)
