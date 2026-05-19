from __future__ import annotations

from pathlib import Path

from pypdf import PdfReader


class PDFReader:
    """SRP: reads a PDF file and returns its plain text."""

    def __init__(self, pdf_path: str) -> None:
        self._path = Path(pdf_path)
        self._reader = PdfReader(str(self._path))

    @property
    def page_count(self) -> int:
        return len(self._reader.pages)

    def extract_text(self) -> str:
        pages = [page.extract_text() or "" for page in self._reader.pages]
        return "\n\n".join(p.strip() for p in pages if p.strip())
