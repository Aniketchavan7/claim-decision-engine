import logging
from typing import List

import bm25s
import Stemmer
import numpy as np

from src.models.evidence import PolicyChunk, RetrievedEvidence

logger = logging.getLogger(__name__)


class SparseRetriever:
    def __init__(self, retriever: bm25s.BM25, chunks: List[PolicyChunk]):
        self.retriever = retriever
        self.chunks = chunks
        self.stemmer = Stemmer.Stemmer("english")

    def search(self, query: str, top_k: int = 25) -> List[RetrievedEvidence]:
        try:
            tokens = bm25s.tokenize(query, stopwords="en", stemmer=self.stemmer)
            docs, scores = self.retriever.retrieve(tokens, k=top_k)

            results = []
            if len(docs) == 0 or len(docs[0]) == 0:
                return results

            for i in range(len(docs[0])):
                doc = docs[0][i]
                score = float(scores[0][i])

                if isinstance(doc, dict) and "id" in doc:
                    chunk = self.chunks[doc["id"]]
                elif isinstance(doc, PolicyChunk):
                    chunk = doc
                elif isinstance(doc, (int, np.integer)):
                    chunk = self.chunks[doc]
                else:
                    chunk = self.chunks[i]

                results.append(
                    RetrievedEvidence(
                        chunk=chunk,
                        score=score,
                        retrieval_method="sparse",
                    )
                )
            return results
        except Exception as e:
            logger.error(f"Error in sparse search: {e}")
            return []
