import logging
from typing import List, Union

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from src.models.evidence import PolicyChunk, RetrievedEvidence

logger = logging.getLogger(__name__)


class DenseRetriever:
    def __init__(
        self,
        index: faiss.Index,
        chunks: List[PolicyChunk],
        model: Union[SentenceTransformer, str] = "BAAI/bge-small-en-v1.5",
        model_name: str = None,
        query_prefix: str = "Represent this sentence for searching relevant passages: ",
    ):
        self.index = index
        self.chunks = chunks
        if isinstance(model, str):
            self.model = SentenceTransformer(model, device="cpu")
        elif model_name:
            self.model = SentenceTransformer(model_name, device="cpu")
        else:
            self.model = model
        self.query_prefix = query_prefix

    def search(self, query: str, top_k: int = 25) -> List[RetrievedEvidence]:
        try:
            full_query = f"{self.query_prefix}{query}"
            query_embedding = self.model.encode([full_query], normalize_embeddings=True)
            query_embedding = np.array(query_embedding, dtype=np.float32)

            distances, indices = self.index.search(query_embedding, top_k)

            results = []
            for i, idx in enumerate(indices[0]):
                if idx < 0 or idx >= len(self.chunks):
                    continue
                score = float(distances[0][i])
                chunk = self.chunks[idx]
                results.append(
                    RetrievedEvidence(
                        chunk=chunk,
                        score=score,
                        retrieval_method="dense",
                    )
                )
            return results
        except Exception as e:
            logger.error(f"Error in dense search: {e}")
            return []
