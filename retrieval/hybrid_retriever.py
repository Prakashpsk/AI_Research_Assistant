"""
retrieval/hybrid_retriever.py
──────────────────────────────
Fuse dense (ChromaDB) and sparse (BM25) results using Reciprocal Rank Fusion (RRF).

Pipeline:
  1. Dense retrieval → ranked chunk_ids
  2. BM25 retrieval → ranked chunk_ids
  3. RRF fusion → top-20 chunk_ids with combined scores
  4. Look up full chunk data from corpus
"""

from __future__ import annotations

from retrieval.vector_store import VectorStore
from retrieval.bm25_retriever import BM25Retriever
from utils.helpers import reciprocal_rank_fusion


class HybridRetriever:
    """
    Combines dense vector search and BM25 keyword search via RRF.
    """

    def __init__(
        self,
        vector_store: VectorStore,
        bm25_retriever: BM25Retriever,
        top_k: int = 20,
        rrf_k: int = 60,
    ):
        self._vector_store = vector_store
        self._bm25 = bm25_retriever
        self.top_k = top_k
        self.rrf_k = rrf_k

    def search(self, query: str, corpus: list[dict]) -> list[dict]:
        """
        Run hybrid retrieval.

        Args:
            query: User question string.
            corpus: Full list of all chunks (used for ID → chunk lookup).

        Returns:
            Top-K chunks sorted by RRF score (descending).
            Each chunk dict has 'dense_score', 'bm25_score', and 'rrf_score' added.
        """
        # Build a lookup map
        id_to_chunk: dict[str, dict] = {c["chunk_id"]: c for c in corpus}

        # Dense retrieval
        dense_results = self._vector_store.search(query, top_k=self.top_k * 2)
        dense_scores: dict[str, float] = {
            r["chunk_id"]: r.get("dense_score", 0.0) for r in dense_results
        }
        dense_ranked = [r["chunk_id"] for r in dense_results]

        # BM25 retrieval
        bm25_results = self._bm25.search(query, top_k=self.top_k * 2)
        bm25_scores: dict[str, float] = {
            r["chunk_id"]: r.get("bm25_score", 0.0) for r in bm25_results
        }
        bm25_ranked = [r["chunk_id"] for r in bm25_results]

        # RRF fusion
        fused = reciprocal_rank_fusion(
            [dense_ranked, bm25_ranked], k=self.rrf_k
        )

        # Build output with all scores
        results = []
        for chunk_id, rrf_score in fused[: self.top_k]:
            chunk = id_to_chunk.get(chunk_id)
            if chunk is None:
                continue
            out = dict(chunk)
            out["dense_score"] = round(dense_scores.get(chunk_id, 0.0), 4)
            out["bm25_score"] = round(bm25_scores.get(chunk_id, 0.0), 4)
            out["rrf_score"] = round(rrf_score, 6)
            results.append(out)

        return results
