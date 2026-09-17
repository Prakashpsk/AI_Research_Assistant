"""
generation/llm_generator.py
────────────────────────────
Answer generation using Google Gemini (gemini-3.8-flash) via google-genai SDK.

Features:
  - Strict grounding: answers ONLY from provided context chunks.
  - Structured JSON output: { answer, citations, refused, reasoning }
  - Anti-hallucination: refuses with explanation if evidence is insufficient.
  - Streaming support for real-time UI updates.
"""

from __future__ import annotations

import json
import os
import re
from typing import Generator

from dotenv import load_dotenv

load_dotenv()

SYSTEM_INSTRUCTION = """You are an Evidence-Grounded AI Research Assistant.
Your ONLY job is to answer questions using EXCLUSIVELY the document chunks provided in the context below.

## STRICT RULES:
1. You MUST NOT use any knowledge from your training data.
2. You MUST cite your sources using the metadata in each chunk header:
   [SOURCE: ...] [PAGE: ...] [CHUNK_ID: ...]
3. If the context does NOT contain sufficient evidence to answer the question, you MUST:
   - Set "refused": true
   - Explain why the evidence is insufficient in the "answer" field
   - Do NOT make up or hallucinate any information
4. Distinguish clearly between direct evidence (what the document says) and inference.
5. Return ONLY valid JSON in this exact schema:

{
  "answer": "<your full answer here, or refusal explanation>",
  "citations": [
    {
      "source": "<filename>",
      "page": <page_number>,
      "chunk_id": "<chunk_id>",
      "quote": "<brief supporting quote from that chunk>"
    }
  ],
  "refused": <true|false>,
  "reasoning": "<brief explanation of how you used the evidence>"
}

Do NOT include any text outside the JSON object."""

MODEL = "gemini-3.8-flash"


class LLMGenerator:
    """
    Generates grounded answers using Gemini with strict citation requirements.
    """

    def __init__(self):
        from google import genai
        api_key = os.getenv("GEMINI_API_KEY")
        self._client = genai.Client(api_key=api_key)

    def generate(
        self,
        query: str,
        context: str,
    ) -> dict:
        """
        Generate a grounded answer from the provided context.

        Args:
            query: User's question.
            context: Formatted context string from ContextBuilder.

        Returns:
            Dict with keys: answer, citations, refused, reasoning.
        """
        if not context.strip():
            return {
                "answer": "No documents have been indexed yet. Please upload and ingest documents first.",
                "citations": [],
                "refused": True,
                "reasoning": "Empty knowledge base.",
            }

        user_input = (
            f"## CONTEXT (retrieved document chunks):\n\n"
            f"{context}\n\n"
            f"## QUESTION:\n{query}\n\n"
            f"Answer the question based ONLY on the context above. "
            f"Return your response as valid JSON matching the required schema."
        )

        try:
            interaction = self._client.interactions.create(
                model=MODEL,
                system_instruction=SYSTEM_INSTRUCTION,
                input=user_input,
                store=False,
            )
            raw = interaction.output_text or ""
            return self._parse_response(raw)
        except Exception as e:
            return {
                "answer": f"LLM error: {e}",
                "citations": [],
                "refused": True,
                "reasoning": str(e),
            }

    def generate_streaming(
        self,
        query: str,
        context: str,
    ) -> Generator[str, None, None]:
        """
        Stream the raw text tokens from Gemini (for Streamlit real-time display).

        Yields raw text deltas. The caller should buffer and parse the full
        JSON at the end.

        Args:
            query: User's question.
            context: Formatted context string from ContextBuilder.

        Yields:
            str text deltas as they arrive.
        """
        if not context.strip():
            yield json.dumps({
                "answer": "No documents indexed. Please upload documents first.",
                "citations": [],
                "refused": True,
                "reasoning": "Empty knowledge base.",
            })
            return

        user_input = (
            f"## CONTEXT (retrieved document chunks):\n\n"
            f"{context}\n\n"
            f"## QUESTION:\n{query}\n\n"
            f"Answer the question based ONLY on the context above. "
            f"Return your response as valid JSON matching the required schema."
        )

        try:
            for event in self._client.interactions.create(
                model=MODEL,
                system_instruction=SYSTEM_INSTRUCTION,
                input=user_input,
                store=False,
                stream=True,
            ):
                if event.event_type == "step.delta":
                    if event.delta.type == "text":
                        yield event.delta.text
        except Exception as e:
            yield json.dumps({
                "answer": f"LLM streaming error: {e}",
                "citations": [],
                "refused": True,
                "reasoning": str(e),
            })

    # ── Internal ──────────────────────────────────────────────────────────────

    @staticmethod
    def _parse_response(raw: str) -> dict:
        """Extract and parse JSON from the model's raw text output."""
        # Strip markdown code fences if present
        raw = raw.strip()
        fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", raw, re.IGNORECASE)
        if fence_match:
            raw = fence_match.group(1).strip()

        try:
            result = json.loads(raw)
            # Ensure required keys
            result.setdefault("answer", "")
            result.setdefault("citations", [])
            result.setdefault("refused", False)
            result.setdefault("reasoning", "")
            return result
        except json.JSONDecodeError:
            # Fallback: treat entire output as plain answer
            return {
                "answer": raw,
                "citations": [],
                "refused": False,
                "reasoning": "Could not parse structured JSON response.",
            }
