"""
retrieval/bm25_retriever.py
────────────────────────────
BM25 keyword-based retrieval using rank_bm25.

The BM25 index is built lazily from the full chunk corpus stored in ChromaDB.
"""

from __future__ import annotations

import re
from typing import Optional

from rank_bm25 import BM25Okapi


class BM25Retriever:
    """
    Sparse keyword retrieval over indexed chunks.
    """

    def __init__(self):
        self._corpus: list[dict] = []
        self._index: Optional[BM25Okapi] = None
        self._tokenized: list[list[str]] = []

    def build_index(self, chunks: list[dict]) -> None:
        """
        Build BM25 index from a list of chunk dicts (must have 'content').

        Args:
            chunks: All chunks in the knowledge base.
        """
        self._corpus = chunks
        self._tokenized = [self._tokenize(c.get("content", "")) for c in chunks]
        self._index = BM25Okapi(self._tokenized)

    def search(self, query: str, top_k: int = 20) -> list[dict]:
        """
        BM25 keyword search.

        Args:
            query: User question.
            top_k: Number of results to return.

        Returns:
            List of chunk dicts with 'bm25_score' added, sorted by score.
        """
        if self._index is None or not self._corpus:
            return []

        tokenized_query = self._tokenize(query)
        scores = self._index.get_scores(tokenized_query)

        # Pair scores with corpus chunks
        scored = [
            (score, chunk)
            for score, chunk in zip(scores, self._corpus)
        ]
        # Sort descending
        scored.sort(key=lambda x: x[0], reverse=True)

        results = []
        for score, chunk in scored[:top_k]:
            c = dict(chunk)
            c["bm25_score"] = float(score)
            results.append(c)

        return results

    def get_ranked_ids(self, query: str, top_k: int = 20) -> list[str]:
        """Return just the ranked list of chunk_ids (for RRF)."""
        return [r["chunk_id"] for r in self.search(query, top_k=top_k)]

    # ── Internal ──────────────────────────────────────────────────────────────

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Lowercase, remove punctuation, split on whitespace."""
        text = text.lower()
        text = re.sub(r"[^\w\s]", " ", text)
        return text.split()
