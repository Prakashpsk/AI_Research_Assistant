"""
retrieval/reranker.py
──────────────────────
Cross-encoder reranking + near-duplicate removal.

Uses: cross-encoder/ms-marco-MiniLM-L-6-v2 (via sentence-transformers)
  - Fast, accurate passage reranking
  - ~22MB model, runs on CPU

Pipeline:
  1. Score each (query, chunk_content) pair with cross-encoder
  2. Sort descending by rerank score
  3. Remove near-duplicate chunks
  4. Return top-k
"""

from __future__ import annotations

from sentence_transformers import CrossEncoder

from utils.helpers import deduplicate_chunks


RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
DEFAULT_TOP_K = 6


class Reranker:
    """
    Cross-encoder reranker with near-duplicate removal.
    """

    def __init__(self, top_k: int = DEFAULT_TOP_K):
        self._model = CrossEncoder(RERANKER_MODEL, max_length=512)
        self.top_k = top_k

    def rerank(self, query: str, chunks: list[dict]) -> list[dict]:
        """
        Rerank retrieved chunks using a cross-encoder.

        Args:
            query: User question.
            chunks: Retrieved chunks from hybrid retriever.

        Returns:
            Top-k reranked, deduplicated chunks with 'rerank_score' added.
        """
        if not chunks:
            return []

        pairs = [(query, c.get("content", "")) for c in chunks]
        scores = self._model.predict(pairs)

        # Attach scores
        for chunk, score in zip(chunks, scores):
            chunk["rerank_score"] = round(float(score), 4)

        # Sort by rerank score descending
        ranked = sorted(chunks, key=lambda x: x.get("rerank_score", 0.0), reverse=True)

        # Deduplicate near-identical chunks
        deduped = deduplicate_chunks(ranked)

        return deduped[: self.top_k]
