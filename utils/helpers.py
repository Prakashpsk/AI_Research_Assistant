"""
utils/helpers.py
────────────────
Shared utility functions used across the RAG pipeline.
"""

import re
import hashlib
import tiktoken

# ── Token Counting ─────────────────────────────────────────────────────────────

_ENCODER = None

def _get_encoder():
    global _ENCODER
    if _ENCODER is None:
        _ENCODER = tiktoken.get_encoding("cl100k_base")
    return _ENCODER


def count_tokens(text: str) -> int:
    """Count approximate tokens using cl100k_base encoding."""
    return len(_get_encoder().encode(text))


# ── Text Cleaning ──────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """
    Normalize whitespace, remove null bytes, collapse multiple blank lines.
    """
    if not text:
        return ""
    text = text.replace("\x00", "")
    text = re.sub(r"\r\n", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ── Chunk ID Generation ────────────────────────────────────────────────────────

def format_chunk_id(source: str, page: int, index: int) -> str:
    """Generate a deterministic chunk ID."""
    raw = f"{source}::p{page}::i{index}"
    return hashlib.md5(raw.encode()).hexdigest()[:12]


# ── Reciprocal Rank Fusion ─────────────────────────────────────────────────────

def reciprocal_rank_fusion(
    ranked_lists: list[list[str]],
    k: int = 60,
) -> list[tuple[str, float]]:
    """
    Fuse multiple ranked lists of chunk IDs using RRF.

    Args:
        ranked_lists: Each inner list is a ranking of chunk_ids (best first).
        k: RRF constant (default 60).

    Returns:
        List of (chunk_id, rrf_score) sorted by descending score.
    """
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, chunk_id in enumerate(ranked, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


# ── Near-Duplicate Detection ───────────────────────────────────────────────────

def deduplicate_chunks(
    chunks: list[dict],
    similarity_threshold: float = 0.85,
) -> list[dict]:
    """
    Remove near-duplicate chunks using character n-gram Jaccard similarity.

    Args:
        chunks: List of chunk dicts (must have 'content' key).
        similarity_threshold: Jaccard threshold above which a chunk is a duplicate.

    Returns:
        Deduplicated list.
    """
    def ngrams(text: str, n: int = 5) -> set:
        return {text[i:i+n] for i in range(len(text) - n + 1)}

    kept = []
    for chunk in chunks:
        content = chunk.get("content", "")
        ng = ngrams(content)
        is_dup = False
        for kept_chunk in kept:
            kept_ng = ngrams(kept_chunk.get("content", ""))
            union = ng | kept_ng
            if not union:
                continue
            jaccard = len(ng & kept_ng) / len(union)
            if jaccard >= similarity_threshold:
                is_dup = True
                break
        if not is_dup:
            kept.append(chunk)
    return kept
