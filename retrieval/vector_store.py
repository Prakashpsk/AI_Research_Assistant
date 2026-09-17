"""
retrieval/vector_store.py
──────────────────────────
Dense semantic retrieval interface wrapping the Embedder's ChromaDB collection.
"""

from __future__ import annotations

from ingestion.embedder import Embedder


class VectorStore:
    """
    Thin wrapper around Embedder providing dense retrieval with ranking info.
    """

    def __init__(self, embedder: Embedder):
        self._embedder = embedder

    def search(self, query: str, top_k: int = 20) -> list[dict]:
        """
        Perform dense semantic search.

        Returns:
            List of chunk dicts sorted by descending cosine similarity.
            Each dict includes 'dense_score' in [0, 1].
        """
        results = self._embedder.query(query, n_results=top_k)
        # Sort descending by score (already sorted by ChromaDB, but be explicit)
        return sorted(results, key=lambda x: x.get("dense_score", 0.0), reverse=True)

    def get_ranked_ids(self, query: str, top_k: int = 20) -> list[str]:
        """Return just the ranked list of chunk_ids (for RRF)."""
        return [r["chunk_id"] for r in self.search(query, top_k=top_k)]
