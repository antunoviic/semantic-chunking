"""
Semantic Chunking — single entry point.

Usage (run from project root):
    python -m app.main <path/to/file.pdf>                        # chunk + evaluate + store
    python -m app.main <path/to/file.pdf> --rechunk              # force re-chunking
    python -m app.main <path/to/file.pdf> --enrich               # with topic/summary enrichment
    python -m app.main <path/to/file.pdf> --query                # open interactive query after eval
    python -m app.main <path/to/file.pdf> --regen-questions      # regenerate QA test set
    python -m app.main <path/to/file.pdf> --top-k 5              # evaluate Hit Rate@5 (default: 3)
    python -m app.main <path/to/file.pdf> --max-questions 20     # limit questions for faster eval
    python -m app.main <path/to/file.pdf> --rag-eval             # end-to-end RAG eval (upload .md to Claude for scoring)
"""

import sys
from pathlib import Path

from .chunking_pipeline import ChunkingPipeline


def _parse_args() -> dict:
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print(__doc__)
        sys.exit(0)

    pdf_path = sys.argv[1]
    if not Path(pdf_path).exists():
        print(f"File not found: {pdf_path}")
        sys.exit(1)

    top_k = 3
    if "--top-k" in sys.argv:
        top_k = int(sys.argv[sys.argv.index("--top-k") + 1])

    max_questions = 50
    if "--max-questions" in sys.argv:
        max_questions = int(sys.argv[sys.argv.index("--max-questions") + 1])

    return {
        "pdf_path":        pdf_path,
        "rechunk":         "--rechunk" in sys.argv,
        "enrich":          "--enrich" in sys.argv,
        "regen_questions": "--regen-questions" in sys.argv,
        "top_k":           top_k,
        "max_questions":   max_questions,
        "query_mode":      "--query" in sys.argv,
        "rag_eval":        "--rag-eval" in sys.argv,
    }


if __name__ == "__main__":
    args = _parse_args()
    pipeline = ChunkingPipeline(
        pdf_path=args["pdf_path"],
        rechunk=args["rechunk"],
        enrich=args["enrich"],
        top_k=args["top_k"],
        max_questions=args["max_questions"],
        regen_questions=args["regen_questions"],
        rag_eval=args["rag_eval"],
    )
    pipeline.run(query_mode=args["query_mode"])
