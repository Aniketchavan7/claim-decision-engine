import logging
from typing import List, Dict

from src.models.evidence import RankedEvidence, DimensionEvidence
from src.retrieval.dense import DenseRetriever
from src.retrieval.sparse import SparseRetriever
from src.retrieval.reranker import Reranker
from src.retrieval.fusion import fuse
from src.config import get_settings

logger = logging.getLogger(__name__)


class HybridRetriever:
    def __init__(self, dense: DenseRetriever, sparse: SparseRetriever, reranker: Reranker, settings=None):
        self.dense = dense
        self.sparse = sparse
        self.reranker = reranker
        self.settings = settings or get_settings()

    def retrieve_single(self, query: str, top_k: int = 10) -> List[RankedEvidence]:
        """Run full pipeline for a single query: dense + sparse + RRF + rerank."""
        try:
            dense_results = self.dense.search(query, top_k=self.settings.dense_top_k)
            sparse_results = self.sparse.search(query, top_k=self.settings.sparse_top_k)

            fused_results = fuse(
                dense_results,
                sparse_results,
                k=self.settings.rrf_k,
                top_k=self.settings.fusion_top_k,
            )

            reranked_results = self.reranker.rerank(query, fused_results, top_k=top_k)
            return reranked_results
        except Exception as e:
            logger.error(f"Error in retrieve_single for query '{query}': {e}")
            return []

    def retrieve_by_dimensions(self, dimensions: Dict[str, List[str]]) -> Dict[str, DimensionEvidence]:
        """For each dimension, run independent retrieval and group results.

        Args:
            dimensions: {dimension_name: [search_queries]}

        Returns:
            {dimension_name: DimensionEvidence}
        """
        dimension_results = {}

        for dimension, queries in dimensions.items():
            all_ranked: List[RankedEvidence] = []
            seen_chunks = set()

            for query in queries:
                results = self.retrieve_single(query, top_k=self.settings.rerank_top_k)
                for res in results:
                    cid = res.chunk.chunk_id
                    if cid not in seen_chunks:
                        seen_chunks.add(cid)
                        all_ranked.append(res)
                    else:
                        for existing in all_ranked:
                            if existing.chunk.chunk_id == cid:
                                if res.final_score > existing.final_score:
                                    existing.final_score = res.final_score
                                break

            all_ranked.sort(key=lambda x: x.final_score, reverse=True)
            top_score = all_ranked[0].final_score if all_ranked else 0.0

            dimension_results[dimension] = DimensionEvidence(
                dimension=dimension,
                queries_used=queries,
                results=all_ranked,
                retrieval_stats={
                    "result_count": len(all_ranked),
                    "top_score": top_score,
                    "queries_count": len(queries),
                },
            )

        return dimension_results
