import logging
from typing import List
from sentence_transformers import CrossEncoder
from src.models.evidence import RankedEvidence

logger = logging.getLogger(__name__)


class Reranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model = CrossEncoder(model_name, device="cpu")

    def rerank(self, query: str, candidates: List[RankedEvidence], top_k: int = 10) -> List[RankedEvidence]:
        if not candidates:
            return []

        try:
            pairs = [[query, c.chunk.text] for c in candidates]
            scores = self.model.predict(pairs)

            for i, c in enumerate(candidates):
                c.rerank_score = float(scores[i])
                c.final_score = float(scores[i])
                c.retrieval_method = "reranked"

            candidates.sort(key=lambda x: x.final_score, reverse=True)
            return candidates[:top_k]
        except Exception as e:
            logger.error(f"Error in reranking: {e}")
            return candidates[:top_k]
