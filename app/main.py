"""
Semantic Chunking — single entry point.

Usage (run from project root):
    python -m app.main <path/to/file.pdf or .txt>                # chunk + evaluate
    python -m app.main <path/to/file.pdf or .txt> --rechunk      # force re-chunking
    python -m app.main <path/to/file.pdf or .txt> --enrich       # with topic enrichment
    python -m app.main <path/to/file.pdf or .txt>                # incremental LLM chunking (sentence-level) — DEFAULT
    python -m app.main <path/to/file.pdf or .txt> --window       # ablation: sliding-window chunking instead
    python -m app.main <path/to/file.pdf or .txt> --step-sentences 2 --max-chunk-sentences 12
                                                                 # finer incremental boundaries / smaller chunks
    python -m app.main <path/to/file.pdf or .txt> --rechunk --max-chunk-chars 700
                                                                 # cap chunk size in CHARS (consistent across docs; needs --rechunk)
    python -m app.main <path/to/file.pdf or .txt> --questions-file eval_cache/foo_questions.json
                                                                 # evaluate against a specific question set (bypasses <stem>_questions.json)
    python -m app.main <path/to/file.pdf or .txt> --max-questions 20 # limit questions for faster eval
"""

import sys
from pathlib import Path

from .chunking_pipeline import ChunkingPipeline


def _parse_args() -> dict:
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print(__doc__)
        sys.exit(0)

    file_path = sys.argv[1]
    if not Path(file_path).exists():
        print(f"File not found: {file_path}")
        sys.exit(1)

    max_questions = 50
    if "--max-questions" in sys.argv:
        max_questions = int(sys.argv[sys.argv.index("--max-questions") + 1])

    step_sentences = 3
    if "--step-sentences" in sys.argv:
        step_sentences = int(sys.argv[sys.argv.index("--step-sentences") + 1])

    max_chunk_sentences = 20
    if "--max-chunk-sentences" in sys.argv:
        max_chunk_sentences = int(sys.argv[sys.argv.index("--max-chunk-sentences") + 1])

    max_chunk_chars = None
    if "--max-chunk-chars" in sys.argv:
        max_chunk_chars = int(sys.argv[sys.argv.index("--max-chunk-chars") + 1])

    questions_file = None
    if "--questions-file" in sys.argv:
        questions_file = sys.argv[sys.argv.index("--questions-file") + 1]
        if not Path(questions_file).exists():
            print(f"Questions file not found: {questions_file}")
            sys.exit(1)

    return {
        "file_path":       file_path,
        "rechunk":         "--rechunk" in sys.argv,
        "enrich":          "--enrich" in sys.argv,
        "incremental":     "--window" not in sys.argv,   # incremental is the default; --window is the ablation
        "step_sentences":  step_sentences,
        "max_chunk_sentences": max_chunk_sentences,
        "max_chunk_chars": max_chunk_chars,
        "questions_file":  questions_file,
        "max_questions":   max_questions,
    }


if __name__ == "__main__":
    args = _parse_args()
    pipeline = ChunkingPipeline(
        file_path=args["file_path"],
        rechunk=args["rechunk"],
        enrich=args["enrich"],
        incremental=args["incremental"],
        step_sentences=args["step_sentences"],
        max_chunk_sentences=args["max_chunk_sentences"],
        max_chunk_chars=args["max_chunk_chars"],
        max_questions=args["max_questions"],
        questions_file=args["questions_file"],
    )
    pipeline.run()
