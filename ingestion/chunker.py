"""
ingestion/chunker.py
──────────────────────
Chunk processed document elements into semantically meaningful pieces.

Strategy:
  - Text elements: split on sentences/paragraphs respecting max_tokens limit.
  - Image elements: one chunk per image (description text).
  - Table elements: one chunk per table (Markdown text).
  - Preserves reading order using bounding box y-coordinate sorting.

Each chunk is a dict:
    {
        "chunk_id":     str,   # deterministic MD5 hash
        "source":       str,   # document filename
        "page":         int,   # page number (1-indexed)
        "element_type": str,   # "text" | "image" | "table"
        "bbox":         list,  # bounding box [x0,y0,x1,y1]
        "content":      str,   # chunk text content
    }
"""

from __future__ import annotations

import re
from typing import Generator

from utils.helpers import count_tokens, clean_text, format_chunk_id


class Chunker:
    """
    Converts a flat list of document elements into overlapping text chunks.
    """

    DEFAULT_MAX_TOKENS = 400
    DEFAULT_OVERLAP_TOKENS = 50

    def __init__(
        self,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
    ):
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens

    def chunk(self, elements: list[dict]) -> list[dict]:
        """
        Convert a list of document elements into a flat list of chunks.

        Args:
            elements: Processed elements from ingestion pipeline.

        Returns:
            List of chunk dicts ready for embedding.
        """
        # Sort by page, then by vertical bbox position (top-to-bottom reading order)
        sorted_elements = sorted(
            elements,
            key=lambda e: (e.get("page", 1), e.get("bbox", [0, 0, 0, 0])[1]),
        )

        chunks = []
        chunk_index = 0

        for element in sorted_elements:
            etype = element.get("element_type", "text")
            content = clean_text(element.get("content", ""))
            source = element.get("source", "unknown")
            page = element.get("page", 1)
            bbox = element.get("bbox", [0.0, 0.0, 0.0, 0.0])

            if not content:
                continue

            if etype in ("image", "table"):
                # Images and tables → single chunk each
                chunk_id = format_chunk_id(source, page, chunk_index)
                chunks.append({
                    "chunk_id": chunk_id,
                    "source": source,
                    "page": page,
                    "element_type": etype,
                    "bbox": bbox,
                    "content": content,
                })
                chunk_index += 1
            else:
                # Text → split with overlap
                for text_chunk in self._split_text(content):
                    chunk_id = format_chunk_id(source, page, chunk_index)
                    chunks.append({
                        "chunk_id": chunk_id,
                        "source": source,
                        "page": page,
                        "element_type": "text",
                        "bbox": bbox,
                        "content": text_chunk,
                    })
                    chunk_index += 1

        return chunks

    # ── Text Splitting ─────────────────────────────────────────────────────────

    def _split_text(self, text: str) -> Generator[str, None, None]:
        """
        Split text into overlapping chunks respecting max_tokens.
        Splits first on paragraph boundaries, then on sentence boundaries.
        """
        sentences = self._split_sentences(text)
        if not sentences:
            return

        current: list[str] = []
        current_tokens = 0
        overlap_buffer: list[str] = []

        for sentence in sentences:
            s_tokens = count_tokens(sentence)

            # If a single sentence exceeds max_tokens, force-split it
            if s_tokens > self.max_tokens:
                if current:
                    yield " ".join(current)
                    overlap_buffer = current[-3:] if len(current) >= 3 else current[:]
                    current = list(overlap_buffer)
                    current_tokens = count_tokens(" ".join(current))

                # Force-split the long sentence by words
                for word_chunk in self._split_by_words(sentence):
                    yield word_chunk
                continue

            if current_tokens + s_tokens > self.max_tokens and current:
                yield " ".join(current)
                # Build overlap from trailing sentences
                overlap_buffer = []
                overlap_tokens = 0
                for sent in reversed(current):
                    tok = count_tokens(sent)
                    if overlap_tokens + tok <= self.overlap_tokens:
                        overlap_buffer.insert(0, sent)
                        overlap_tokens += tok
                    else:
                        break
                current = overlap_buffer
                current_tokens = overlap_tokens

            current.append(sentence)
            current_tokens += s_tokens

        if current:
            yield " ".join(current)

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        """Split text into sentences using punctuation heuristics."""
        # Split on . ! ? followed by whitespace or end-of-string
        parts = re.split(r"(?<=[.!?])\s+", text)
        # Further split on double newlines (paragraph breaks)
        result = []
        for part in parts:
            sub = [p.strip() for p in part.split("\n\n") if p.strip()]
            result.extend(sub)
        return [s for s in result if s]

    def _split_by_words(self, text: str) -> Generator[str, None, None]:
        """Force-split a very long string by word boundaries."""
        words = text.split()
        current: list[str] = []
        current_tokens = 0
        for word in words:
            w_tokens = count_tokens(word)
            if current_tokens + w_tokens > self.max_tokens and current:
                yield " ".join(current)
                current = []
                current_tokens = 0
            current.append(word)
            current_tokens += w_tokens
        if current:
            yield " ".join(current)
