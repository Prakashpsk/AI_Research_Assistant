"""
ingestion/table_extractor.py
──────────────────────────────
Convert raw table elements (from document_parser) into Markdown-formatted text
using PyMuPDF's find_tables() cell extraction.
"""

from __future__ import annotations


class TableExtractor:
    """
    Converts PyMuPDF table objects to Markdown string format.
    """

    def process(self, element: dict) -> dict:
        """
        Fill the 'content' field of a table element with Markdown.

        Args:
            element: Raw table element dict from DocumentParser (element_type='table').

        Returns:
            Same element dict with 'content' populated.
        """
        raw = element.get("raw", {})
        table_obj = raw.get("table_obj")

        if table_obj is None:
            element["content"] = ""
            return element

        try:
            md = self._table_to_markdown(table_obj)
        except Exception as e:
            md = f"[Table extraction error: {e}]"

        element["content"] = md
        return element

    def process_all(self, elements: list[dict]) -> list[dict]:
        """Process all table elements in a list, leaving others unchanged."""
        return [
            self.process(el) if el.get("element_type") == "table" else el
            for el in elements
        ]

    # ── Internal ──────────────────────────────────────────────────────────────

    @staticmethod
    def _table_to_markdown(table_obj) -> str:
        """
        Convert a fitz Table object to GitHub-flavored Markdown.
        """
        try:
            rows = table_obj.extract()  # list of list of str
        except Exception:
            return "[Could not extract table cells]"

        if not rows:
            return ""

        # Clean cells
        cleaned = []
        for row in rows:
            cleaned_row = [
                str(cell).replace("\n", " ").strip() if cell is not None else ""
                for cell in row
            ]
            cleaned.append(cleaned_row)

        # Determine column widths
        num_cols = max(len(row) for row in cleaned)
        col_widths = [0] * num_cols
        for row in cleaned:
            for col_idx, cell in enumerate(row):
                if col_idx < num_cols:
                    col_widths[col_idx] = max(col_widths[col_idx], len(cell))

        def format_row(cells: list[str]) -> str:
            padded = []
            for i in range(num_cols):
                val = cells[i] if i < len(cells) else ""
                padded.append(val.ljust(col_widths[i]))
            return "| " + " | ".join(padded) + " |"

        def separator() -> str:
            return "| " + " | ".join("-" * max(w, 3) for w in col_widths) + " |"

        lines = []
        if cleaned:
            lines.append(format_row(cleaned[0]))  # header
            lines.append(separator())
            for row in cleaned[1:]:
                lines.append(format_row(row))

        return "\n".join(lines)
