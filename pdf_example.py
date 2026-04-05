import sys
from pathlib import Path
from pypdf import PdfReader

from llm_chunker import LLMChunker, QwenClient


def extract_text_from_pdf(pdf_path: str) -> str:
    reader = PdfReader(pdf_path)
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(p.strip() for p in pages if p.strip())


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python pdf_example.py <path/to/file.pdf>")
        sys.exit(1)

    pdf_path = sys.argv[1]
    if not Path(pdf_path).exists():
        print(f"File not found: {pdf_path}")
        sys.exit(1)

    print(f"Reading: {pdf_path}")
    text = extract_text_from_pdf(pdf_path)
    print(f"Extracted {len(text)} characters from {len(PdfReader(pdf_path).pages)} pages\n")

    chunker = LLMChunker(client=QwenClient())
    chunks = chunker.chunk(text)

    print(f"{len(chunks)} Chunks found:\n")
    for i, chunk in enumerate(chunks, 1):
        print(f"--- Chunk {i} ---")
        print(chunk)
        print()
