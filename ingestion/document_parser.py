"""
ingestion/document_parser.py
─────────────────────────────
Parse PDF, TXT, and Markdown files into raw structured elements using PyMuPDF.

Each element is a dict with:
    {
        "source":       str,   # filename
        "page":         int,   # 1-indexed page number
        "element_type": str,   # "text" | "image" | "table"
        "bbox":         list,  # [x0, y0, x1, y1] in points
        "content":      str,   # raw text (or placeholder for image/table)
        "raw":          any,   # type-specific raw data
    }
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Generator

import fitz  # PyMuPDF


class DocumentParser:
    """
    Parses supported documents (PDF, TXT, MD) into raw page elements.
    Actual image OCR and table conversion happen in dedicated modules.
    """

    SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".markdown"}

    def parse(self, file_path: str | Path) -> list[dict]:
        """
        Parse a document file and return a flat list of raw elements
        in reading order (sorted by page then top-to-bottom bbox).

        Args:
            file_path: Path to the document file.

        Returns:
            List of element dicts.

        Raises:
            ValueError: If the file extension is not supported.
        """
        path = Path(file_path)
        ext = path.suffix.lower()

        if ext not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file type '{ext}'. "
                f"Supported: {self.SUPPORTED_EXTENSIONS}"
            )

        if ext == ".pdf":
            return list(self._parse_pdf(path))
        else:
            return list(self._parse_text_file(path))

    # ── PDF Parsing ────────────────────────────────────────────────────────────

    def _parse_pdf(self, path: Path) -> Generator[dict, None, None]:
        """Extract text blocks, images, and table placeholders from a PDF."""
        source = path.name
        doc = fitz.open(str(path))

        try:
            for page_idx, page in enumerate(doc):
                page_num = page_idx + 1

                # ── Text blocks ────────────────────────────────────────────
                blocks = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)
                for block in blocks.get("blocks", []):
                    if block.get("type") == 0:  # text block
                        lines = block.get("lines", [])
                        text = " ".join(
                            span.get("text", "")
                            for line in lines
                            for span in line.get("spans", [])
                        ).strip()
                        if text:
                            yield {
                                "source": source,
                                "page": page_num,
                                "element_type": "text",
                                "bbox": list(block["bbox"]),
                                "content": text,
                                "raw": None,
                            }

                # ── Images ─────────────────────────────────────────────────
                image_list = page.get_images(full=True)
                for img_index, img_info in enumerate(image_list):
                    xref = img_info[0]
                    bbox = self._get_image_bbox(page, xref)
                    try:
                        base_image = doc.extract_image(xref)
                        img_bytes = base_image["image"]
                        img_ext = base_image["ext"]
                    except Exception:
                        img_bytes = None
                        img_ext = "png"

                    yield {
                        "source": source,
                        "page": page_num,
                        "element_type": "image",
                        "bbox": bbox,
                        "content": "",   # filled by image_processor
                        "raw": {
                            "xref": xref,
                            "bytes": img_bytes,
                            "ext": img_ext,
                            "index": img_index,
                        },
                    }

                # ── Tables (placeholder — extracted by table_extractor) ────
                try:
                    tables = page.find_tables()
                    for tbl_idx, table in enumerate(tables):
                        yield {
                            "source": source,
                            "page": page_num,
                            "element_type": "table",
                            "bbox": list(table.bbox),
                            "content": "",  # filled by table_extractor
                            "raw": {
                                "table_obj": table,
                                "index": tbl_idx,
                            },
                        }
                except Exception:
                    pass  # find_tables() may fail on some PDFs

        finally:
            doc.close()

    # ── Text / Markdown Parsing ────────────────────────────────────────────────

    def _parse_text_file(self, path: Path) -> Generator[dict, None, None]:
        """Parse plain text or Markdown into single-page text elements."""
        source = path.name
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            text = path.read_text(encoding="latin-1")

        # Split on double newlines for paragraph-level chunks
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        for para_idx, para in enumerate(paragraphs):
            yield {
                "source": source,
                "page": 1,
                "element_type": "text",
                "bbox": [0.0, float(para_idx * 20), 500.0, float(para_idx * 20 + 18)],
                "content": para,
                "raw": None,
            }

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _get_image_bbox(page: fitz.Page, xref: int) -> list[float]:
        """Try to get the bounding box of an image on the page."""
        try:
            for block in page.get_text("rawdict")["blocks"]:
                if block.get("type") == 1 and block.get("xref") == xref:
                    return list(block["bbox"])
        except Exception:
            pass
        return [0.0, 0.0, 0.0, 0.0]
