import pytest
from src.models.evidence import PolicyChunk, RetrievedEvidence
from src.retrieval.fusion import fuse


def test_rrf_fusion():
    c1 = PolicyChunk(chunk_id="c1", text="Hospitalization cover", page=1)
    c2 = PolicyChunk(chunk_id="c2", text="Exclusion 4.1", page=3)
    c3 = PolicyChunk(chunk_id="c3", text="Waiting period", page=5)

    dense = [
        RetrievedEvidence(chunk=c1, score=0.9, retrieval_method="dense"),
        RetrievedEvidence(chunk=c2, score=0.7, retrieval_method="dense"),
    ]
    sparse = [
        RetrievedEvidence(chunk=c2, score=12.0, retrieval_method="sparse"),
        RetrievedEvidence(chunk=c3, score=8.0, retrieval_method="sparse"),
    ]

    fused = fuse(dense, sparse, k=60, top_k=3)
    assert len(fused) == 3
    # c2 was ranked 2nd in dense and 1st in sparse, should have high RRF score
    assert fused[0].chunk.chunk_id == "c2"
