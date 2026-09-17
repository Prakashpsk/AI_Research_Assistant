"""
generation/context_builder.py
───────────────────────────────
Build a safe, structured context block from reranked chunks for LLM prompting.

Security features:
  - Wraps each chunk in [UNTRUSTED_DOCUMENT_CONTENT] tags
  - Sanitizes prompt-injection patterns
  - Adds strict boundaries to prevent context leakage

Output format:
  Each chunk is rendered as:

    [SOURCE: {source} | PAGE: {page} | CHUNK_ID: {chunk_id} | TYPE: {element_type}]
    [BEGIN_UNTRUSTED]
    {content}
    [END_UNTRUSTED]
"""

from __future__ import annotations

import re


# Patterns that suggest prompt injection
_INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?previous\s+instructions?",
    r"disregard\s+(all\s+)?previous\s+instructions?",
    r"you\s+are\s+now\s+(?:a\s+)?(?:different|new|evil|jailbreak)",
    r"forget\s+(?:all\s+)?(?:your\s+)?(?:previous\s+)?instructions?",
    r"system\s*:\s*you\s+are",
    r"act\s+as\s+(?:if\s+)?(?:you\s+are\s+)?(?:an?\s+)?(?:unrestricted|unfiltered|jailbreak)",
    r"\[INST\]",
    r"<\|system\|>",
    r"<\|user\|>",
    r"<\|assistant\|>",
]

_INJECTION_RE = re.compile(
    "|".join(_INJECTION_PATTERNS), re.IGNORECASE | re.MULTILINE
)


class ContextBuilder:
    """
    Builds a structured, injection-safe context string from ranked chunks.
    """

    MAX_CONTEXT_CHARS = 30_000  # ~7-8k tokens safety limit

    def build(self, chunks: list[dict]) -> tuple[str, list[dict]]:
        """
        Build the context block for the LLM prompt.

        Args:
            chunks: Reranked chunk dicts.

        Returns:
            (context_str, included_chunks)
            - context_str: Formatted string to inject into the prompt.
            - included_chunks: Subset of chunks actually included (may be truncated).
        """
        sections: list[str] = []
        included: list[dict] = []
        total_chars = 0

        for i, chunk in enumerate(chunks, start=1):
            content = self._sanitize(chunk.get("content", ""))
            if not content:
                continue

            section = (
                f"[SOURCE: {chunk.get('source', 'unknown')} | "
                f"PAGE: {chunk.get('page', '?')} | "
                f"CHUNK_ID: {chunk.get('chunk_id', '?')} | "
                f"TYPE: {chunk.get('element_type', 'text')}]\n"
                f"[BEGIN_UNTRUSTED]\n"
                f"{content}\n"
                f"[END_UNTRUSTED]"
            )

            section_chars = len(section)
            if total_chars + section_chars > self.MAX_CONTEXT_CHARS:
                break

            sections.append(f"--- Chunk {i} ---\n{section}")
            included.append(chunk)
            total_chars += section_chars

        context_str = "\n\n".join(sections) if sections else ""
        return context_str, included

    # ── Internal ──────────────────────────────────────────────────────────────

    @staticmethod
    def _sanitize(text: str) -> str:
        """Remove prompt injection patterns from chunk content."""
        return _INJECTION_RE.sub("[REDACTED]", text)
