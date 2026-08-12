from __future__ import annotations

import re
from pathlib import Path

from pypdf import PdfReader


class DocumentReader:
    """SRP: reads a document file (.pdf or .txt) and returns its plain text."""

    SUPPORTED_EXTENSIONS = {".pdf", ".txt"}

    def __init__(self, file_path: str) -> None:
        self._path = Path(file_path)
        suffix = self._path.suffix.lower()
        if suffix not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file type: '{suffix}'. "
                f"Supported: {', '.join(sorted(self.SUPPORTED_EXTENSIONS))}"
            )
        self._is_pdf = suffix == ".pdf"
        self._reader = PdfReader(str(self._path)) if self._is_pdf else None

    @property
    def page_count(self) -> int:
        return len(self._reader.pages) if self._is_pdf else 1

    def extract_text(self) -> str:
        if self._is_pdf:
            pages = [self._dewrap(page.extract_text() or "") for page in self._reader.pages]
            return "\n\n".join(p for p in pages if p)
        return self._path.read_text(encoding="utf-8").strip()

    @staticmethod
    def _dewrap(text: str) -> str:
        r"""Undo pypdf's mid-paragraph line breaks.

        PDF extraction inserts a newline at every visual line, so words that
        merely wrapped to the next line get split by '\n' (sometimes with a
        hyphen). This joins them back together while keeping real paragraph
        breaks (blank lines):
          "manag -\ning" -> "managing"    (de-hyphenated word wrap)
          "word1\nword2" -> "word1 word2"  (single newline -> space)
          blank line     -> paragraph break (kept)
        """
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        # de-hyphenate words split across a line break
        text = re.sub(r"(?<=\w) ?- ?\n ?(?=\w)", "", text)
        # split into paragraphs on blank lines; within each, join wrapped lines
        paragraphs = re.split(r"\n[ \t]*\n+", text)
        cleaned = []
        for para in paragraphs:
            joined = re.sub(r"[ \t]*\n[ \t]*", " ", para)   # single newlines -> spaces
            joined = re.sub(r"[ \t]{2,}", " ", joined).strip()
            if joined:
                cleaned.append(joined)
        return "\n\n".join(cleaned)
