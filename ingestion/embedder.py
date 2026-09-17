"""
ingestion/embedder.py
──────────────────────
Embed chunks using BGE-small-en-v1.5 and store them in ChromaDB.

Responsibilities:
  - Load BGE-small-en-v1.5 model via sentence-transformers.
  - Batch-embed chunk content.
  - Persist embeddings + metadata to ChromaDB collection.
  - Provide add / query / delete / list operations.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import chromadb
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


CHROMA_DB_PATH = str(Path(__file__).resolve().parent.parent / "data" / "chroma_db")
COLLECTION_NAME = "research_chunks"
EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"
BATCH_SIZE = 64


class Embedder:
    """
    Manages BGE embeddings and ChromaDB persistence for document chunks.
    """

    def __init__(self, db_path: str = CHROMA_DB_PATH):
        abs_db_path = str(Path(db_path).resolve())
        os.makedirs(abs_db_path, exist_ok=True)
        self._model = SentenceTransformer(EMBED_MODEL_NAME)
        self._client = chromadb.PersistentClient(path=abs_db_path)
        self._collection = self._client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    # ── Indexing ───────────────────────────────────────────────────────────────

    def add_chunks(self, chunks: list[dict], show_progress: bool = True) -> int:
        """
        Embed and add chunks to ChromaDB.

        Skips chunks whose chunk_id already exists in the collection.

        Args:
            chunks: List of chunk dicts (must have 'chunk_id', 'content', metadata).
            show_progress: Show tqdm progress bar.

        Returns:
            Number of new chunks actually added.
        """
        if not chunks:
            return 0

        # Filter out already-indexed chunks
        existing_ids = set(self._collection.get(include=[])["ids"])
        new_chunks = [c for c in chunks if c["chunk_id"] not in existing_ids]

        if not new_chunks:
            return 0

        added = 0
        iterator = range(0, len(new_chunks), BATCH_SIZE)
        if show_progress:
            iterator = tqdm(iterator, desc="Embedding & indexing", unit="batch")

        for batch_start in iterator:
            batch = new_chunks[batch_start : batch_start + BATCH_SIZE]
            contents = [c["content"] for c in batch]

            # BGE prefix for retrieval task
            prefixed = [f"Represent this sentence: {t}" for t in contents]
            embeddings = self._model.encode(
                prefixed, normalize_embeddings=True
            ).tolist()

            ids = [c["chunk_id"] for c in batch]
            metadatas = [
                {
                    "source": c.get("source", ""),
                    "page": c.get("page", 0),
                    "element_type": c.get("element_type", "text"),
                    "bbox": str(c.get("bbox", [])),
                }
                for c in batch
            ]
            documents = contents

            self._collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas,
            )
            added += len(batch)

        return added

    # ── Querying ───────────────────────────────────────────────────────────────

    def query(self, query_text: str, n_results: int = 20) -> list[dict]:
        """
        Dense semantic search using BGE embeddings.

        Args:
            query_text: User question.
            n_results: Number of results to return.

        Returns:
            List of result dicts with chunk_id, content, score, and metadata.
        """
        query_prefixed = f"Represent this sentence: {query_text}"
        query_embedding = self._model.encode(
            [query_prefixed], normalize_embeddings=True
        ).tolist()

        total = self._collection.count()
        if total == 0:
            return []
        n_results = min(n_results, total)

        results = self._collection.query(
            query_embeddings=query_embedding,
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )

        output = []
        ids = results["ids"][0]
        docs = results["documents"][0]
        metas = results["metadatas"][0]
        dists = results["distances"][0]

        for chunk_id, doc, meta, dist in zip(ids, docs, metas, dists):
            output.append({
                "chunk_id": chunk_id,
                "content": doc,
                "source": meta.get("source", ""),
                "page": meta.get("page", 0),
                "element_type": meta.get("element_type", "text"),
                "bbox": meta.get("bbox", "[]"),
                "dense_score": float(1.0 - dist),  # cosine similarity
            })

        return output

    # ── Collection Management ─────────────────────────────────────────────────

    def count(self) -> int:
        """Return total number of indexed chunks."""
        return self._collection.count()

    def list_sources(self) -> list[str]:
        """Return list of unique document sources indexed."""
        if self._collection.count() == 0:
            return []
        all_meta = self._collection.get(include=["metadatas"])["metadatas"]
        sources = sorted({m.get("source", "") for m in all_meta if m.get("source")})
        return sources

    def delete_source(self, source_name: str) -> int:
        """Delete all chunks from a given source document."""
        all_data = self._collection.get(include=["metadatas"])
        ids_to_delete = [
            cid
            for cid, meta in zip(all_data["ids"], all_data["metadatas"])
            if meta.get("source") == source_name
        ]
        if ids_to_delete:
            self._collection.delete(ids=ids_to_delete)
        return len(ids_to_delete)

    def get_all_chunks(self) -> list[dict]:
        """Return all chunks stored in ChromaDB (for BM25 index building)."""
        if self._collection.count() == 0:
            return []
        data = self._collection.get(include=["documents", "metadatas"])
        chunks = []
        for cid, doc, meta in zip(data["ids"], data["documents"], data["metadatas"]):
            chunks.append({
                "chunk_id": cid,
                "content": doc,
                "source": meta.get("source", ""),
                "page": meta.get("page", 0),
                "element_type": meta.get("element_type", "text"),
                "bbox": meta.get("bbox", "[]"),
            })
        return chunks
