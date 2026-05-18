"""
PDF → Chunks → ChromaDB → Query

Usage:
    python pdf_example.py <path/to/file.pdf>           # load from cache if available
    python pdf_example.py <path/to/file.pdf> --rechunk  # force re-chunking

Chunk cache is saved to ./chunks_cache/<filename>.json
Old chunks are backed up as <filename>.<timestamp>.json for comparison.
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional
from pypdf import PdfReader

from llm_chunker import LLMChunker, QwenClient
from llm_chunker.vectorstore import VectorStore

CACHE_DIR = Path("./chunks_cache")


def extract_text_from_pdf(pdf_path: str) -> str:
    reader = PdfReader(pdf_path)
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(p.strip() for p in pages if p.strip())


def cache_path(pdf_path: str, enriched: bool = False) -> Path:
    suffix = "_enriched" if enriched else ""
    return CACHE_DIR / (Path(pdf_path).stem + suffix + ".json")


def load_chunks(pdf_path: str, enriched: bool = False) -> Optional[list[str]]:
    path = cache_path(pdf_path, enriched)
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        print(f"[cache] Loaded {len(data['chunks'])} chunks from {path}")
        print(f"        Chunked on: {data['chunked_at']}")
        return data["chunks"]
    return None


def save_chunks(pdf_path: str, chunks: list[str], enriched: bool = False) -> None:
    CACHE_DIR.mkdir(exist_ok=True)
    path = cache_path(pdf_path, enriched)

    # Back up existing cache before overwriting
    if path.exists():
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup = path.with_suffix(f".{ts}.json")
        path.rename(backup)
        print(f"[cache] Old chunks backed up to {backup.name}")

    data = {
        "source": Path(pdf_path).name,
        "chunked_at": datetime.now().isoformat(),
        "chunk_count": len(chunks),
        "chunks": chunks,
    }
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[cache] Saved {len(chunks)} chunks to {path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python pdf_example.py <path/to/file.pdf> [--rechunk] [--enrich]")
        sys.exit(1)

    pdf_path      = sys.argv[1]
    force_rechunk = "--rechunk" in sys.argv
    use_enrich    = "--enrich" in sys.argv

    if not Path(pdf_path).exists():
        print(f"File not found: {pdf_path}")
        sys.exit(1)

    # --- Step 1: Read PDF ---
    print(f"Reading: {pdf_path}")
    text = extract_text_from_pdf(pdf_path)
    print(f"Extracted {len(text)} chars from {len(PdfReader(pdf_path).pages)} pages\n")

    # --- Step 2: Chunk (or load from cache) ---
    chunks = None if force_rechunk else load_chunks(pdf_path, enriched=use_enrich)

    if chunks is None:
        chunker = LLMChunker(
            client=QwenClient(),
            window_size=10,
            sentences_per_mini_chunk=3,
            filter_low_info=True,
            enrich=use_enrich,
            verbose=True,
        )
        chunks = chunker.chunk(text)
        save_chunks(pdf_path, chunks, enriched=use_enrich)

    print(f"\n{'='*60}")
    print(f"{len(chunks)} semantic chunks found:\n")
    for i, chunk in enumerate(chunks, 1):
        print(f"--- Chunk {i} ({len(chunk)} chars) ---")
        print(chunk[:300] + ("..." if len(chunk) > 300 else ""))
        print()

    # --- Step 3: Store in ChromaDB ---
    print("Storing chunks in ChromaDB...")
    store = VectorStore(persist_dir="./chroma_db")
    store.add_chunks("llm_semantic", chunks, source=Path(pdf_path).name)

    # --- Step 4: Interactive query loop ---
    print(f"\n{'='*60}")
    print("Ask questions about the document. Type 'quit' to exit.\n")

    while True:
        query = input("Question: ").strip()
        if not query or query.lower() in ("quit", "exit", "q"):
            break

        results = store.query("llm_semantic", query, k=3)
        print()
        for i, r in enumerate(results, 1):
            preview = r.chunk_text[:200].replace("\n", " ")
            print(f"  [{i}] distance={r.distance:.3f}")
            print(f"      {preview}...")
            print()