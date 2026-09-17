"""
ingestion/image_processor.py
──────────────────────────────
Process image elements by:
  1. Encoding image bytes as base64.
  2. Calling Gemini multimodal (gemini-3.8-flash) to generate a rich description /
     OCR of the image content (charts, diagrams, scanned text, etc.).

No PaddleOCR dependency — Gemini vision handles all image understanding.
"""

from __future__ import annotations

import base64
import os
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


class ImageProcessor:
    """
    Uses Gemini multimodal to describe / OCR images extracted from documents.
    Falls back to a placeholder if the API call fails.
    """

    IMAGE_PROMPT = (
        "You are an AI assistant processing an image extracted from a research document. "
        "Describe the image content in detail:\n"
        "- If it contains text, transcribe all readable text verbatim.\n"
        "- If it is a chart or graph, describe the axes, data series, trends, and key values.\n"
        "- If it is a diagram or figure, describe what it depicts and any labels.\n"
        "- If it is a table (in image form), transcribe it as a Markdown table.\n"
        "Be thorough and factual. Do not add interpretations beyond what is visible."
    )

    def __init__(self):
        from google import genai
        api_key = os.getenv("GEMINI_API_KEY")
        self._client = genai.Client(api_key=api_key)

    def process(self, element: dict) -> dict:
        """
        Fill the 'content' field of an image element with Gemini's description.

        Args:
            element: Raw image element dict (element_type='image').

        Returns:
            Same element dict with 'content' populated.
        """
        raw = element.get("raw", {})
        img_bytes = raw.get("bytes")
        img_ext = raw.get("ext", "png")

        if not img_bytes:
            element["content"] = "[Image: no bytes extracted]"
            return element

        try:
            description = self._describe_image(img_bytes, img_ext)
        except Exception as e:
            description = f"[Image description unavailable: {e}]"

        element["content"] = description
        return element

    def process_all(self, elements: list[dict]) -> list[dict]:
        """Process all image elements; leave others unchanged."""
        results = []
        for el in elements:
            if el.get("element_type") == "image":
                results.append(self.process(el))
                time.sleep(0.3)  # small delay to avoid rate limiting
            else:
                results.append(el)
        return results

    # ── Internal ──────────────────────────────────────────────────────────────

    def _describe_image(self, img_bytes: bytes, ext: str) -> str:
        """Call Gemini with the image bytes and return a description string."""
        mime_map = {
            "png": "image/png",
            "jpg": "image/jpeg",
            "jpeg": "image/jpeg",
            "gif": "image/gif",
            "webp": "image/webp",
            "bmp": "image/bmp",
        }
        mime_type = mime_map.get(ext.lower(), "image/png")
        b64_data = base64.b64encode(img_bytes).decode("utf-8")

        from google.genai import types

        interaction = self._client.interactions.create(
            model="gemini-3.8-flash",
            input=[
                types.Part.from_bytes(data=base64.b64decode(b64_data), mime_type=mime_type),
                self.IMAGE_PROMPT,
            ],
            store=False,
        )
        return interaction.output_text or "[Image: no description generated]"
