"""
    python -m app.main <path/to/file.pdf or .txt>                # chunk + evaluate
    python -m app.main <path/to/file.pdf or .txt> --rechunk      # force re-chunking
    python -m app.main <path/to/file.pdf or .txt> --enrich       # with topic enrichment
    python -m app.main <path/to/file.pdf or .txt>                # incremental LLM chunking - default
    python -m app.main <path/to/file.pdf or .txt> --window       # sliding-window chunking in fixed sizes
    python -m app.main <path/to/file.pdf or .txt> --step-sentences 2 --max-chunk-sentences 12
                                                                 # finer incremental boundaries / smaller chunks
    python -m app.main <path/to/file.pdf or .txt> --rechunk --max-chunk-chars 700
                                                                 # cap chunk size in chars
    python -m app.main <path/to/file.pdf or .txt> --no-headings  # ablation: disable heading-aware splitting
    python -m app.main <path/to/file.pdf or .txt> --llm-headings
                                                                 # heading detection: strong line-based regex
                                                                 # + LLM fallback where the regex finds nothing
                                                                 # (default without this flag: regex-only, shipped)
    python -m app.main <path/to/file.pdf or .txt> --chunk-only   # only chunk+cache (no question set needed yet)
    python -m app.main <path/to/file.pdf or .txt> --no-filter      # ablation: keep low-info chunks
    python -m app.main <path/to/file.pdf or .txt> --midpoint-split
                                                                 # ablation: split in the middle instead of asking the LLM
    python -m app.main <path/to/file.pdf or .txt> --questions-file eval_cache/foo_questions.json
                                                                 # evaluate against a specific question set(natural/literal)
    python -m app.main <path/to/file.pdf or .txt> --max-questions 20 # limit questions
"""

import sys
from pathlib import Path

from .pipeline import Pipeline


def _parse_args() -> dict:
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print(__doc__)
        sys.exit(0)

    file_path = sys.argv[1]
    if not Path(file_path).exists():
        print(f"File not found: {file_path}")
        sys.exit(1)

    max_questions = 1000         
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
        "respect_headings": "--no-headings" not in sys.argv,  # heading-aware split is on by default
        "heading_mode":    "hybrid" if "--llm-headings" in sys.argv else "regex",
        "smart_split":     "--midpoint-split" not in sys.argv,
        "filter_low_info": "--no-filter" not in sys.argv,      # Ablation: LowInfoFilter
        "chunk_only":      "--chunk-only" in sys.argv,        # only chunk+cache, skip evaluation
        "questions_file":  questions_file,
        "max_questions":   max_questions,
    }


if __name__ == "__main__":
    args = _parse_args()
    pipeline = Pipeline(
        file_path=args["file_path"],
        rechunk=args["rechunk"],
        enrich=args["enrich"],
        incremental=args["incremental"],
        step_sentences=args["step_sentences"],
        max_chunk_sentences=args["max_chunk_sentences"],
        max_chunk_chars=args["max_chunk_chars"],
        respect_headings=args["respect_headings"],
        heading_mode=args["heading_mode"],
        smart_split=args["smart_split"],
        filter_low_info=args["filter_low_info"],
        chunk_only=args["chunk_only"],
        max_questions=args["max_questions"],
        questions_file=args["questions_file"],
    )
    pipeline.run()
