from typing import List, Dict
from src.models.evidence import RetrievedEvidence, RankedEvidence


def fuse(
    dense_results: List[RetrievedEvidence],
    sparse_results: List[RetrievedEvidence],
    k: int = 60,
    top_k: int = 15,
) -> List[RankedEvidence]:
    rrf_scores: Dict[str, float] = {}
    dense_ranks: Dict[str, int] = {}
    sparse_ranks: Dict[str, int] = {}
    chunk_map: Dict[str, RetrievedEvidence] = {}

    for rank, item in enumerate(dense_results):
        cid = item.chunk.chunk_id
        if cid not in chunk_map:
            chunk_map[cid] = item
        dense_ranks[cid] = rank + 1
        rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (k + rank + 1)

    for rank, item in enumerate(sparse_results):
        cid = item.chunk.chunk_id
        if cid not in chunk_map:
            chunk_map[cid] = item
        sparse_ranks[cid] = rank + 1
        rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (k + rank + 1)

    ranked = []
    for chunk_id, score in rrf_scores.items():
        evidence = chunk_map[chunk_id]
        ranked.append(
            RankedEvidence(
                chunk=evidence.chunk,
                final_score=score,
                dense_score=evidence.score if evidence.retrieval_method == "dense" else 0.0,
                sparse_score=evidence.score if evidence.retrieval_method == "sparse" else 0.0,
                dense_rank=dense_ranks.get(chunk_id, 0),
                sparse_rank=sparse_ranks.get(chunk_id, 0),
                retrieval_method="hybrid_rrf",
            )
        )

    ranked.sort(key=lambda x: x.final_score, reverse=True)
    return ranked[:top_k]
