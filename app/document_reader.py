from __future__ import annotations

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
            pages = [page.extract_text() or "" for page in self._reader.pages]
            return "\n\n".join(p.strip() for p in pages if p.strip())
        return self._path.read_text(encoding="utf-8").strip()
