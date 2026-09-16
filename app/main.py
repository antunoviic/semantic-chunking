from __future__ import annotations

import argparse
from pathlib import Path

from llm_semantic_chunker import ChunkerConfig

from .pipeline import EvalConfig, Pipeline


def _existing_file(value: str) -> str:
    if not Path(value).exists():
        raise argparse.ArgumentTypeError(f"file not found: {value}")
    return value


def _positive_int(value: str) -> int:
    try:
        number = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected an integer, got {value!r}") from None
    if number < 1:
        raise argparse.ArgumentTypeError(f"must be >= 1, got {number}")
    return number


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m app.main",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("document", type=_existing_file,
                        help="the .pdf or .txt to chunk")

    run = parser.add_argument_group("what to run")
    run.add_argument("--chunk-only", action="store_true",
                     help="only chunk and cache; skip retrieval (no question set needed)")
    run.add_argument("--rechunk", action="store_true",
                     help="ignore the cache and chunk again")
    run.add_argument("--questions-file", type=_existing_file, metavar="PATH",
                     help="question set to evaluate against; defaults to "
                          "eval_cache/<document>_questions.json")
    run.add_argument("--max-questions", type=_positive_int, default=1000, metavar="N",
                     help="evaluate at most N questions (default: all)")

    mode = parser.add_argument_group("chunking mode")
    mode.add_argument("--window", action="store_true",
                      help="sliding-window chunking instead of the incremental "
                           "default; kept for comparison, collapses without a size cap")
    mode.add_argument("--step-sentences", type=_positive_int, default=3, metavar="N",
                      help="sentences per boundary decision (default: 3)")
    mode.add_argument("--max-chunk-sentences", type=_positive_int, default=20, metavar="N",
                      help="sentence cap regardless of topic continuity (default: 20)")
    mode.add_argument("--max-chunk-chars", type=_positive_int, default=None, metavar="N",
                      help="character cap, applied at sentence boundaries; a single "
                           "sentence longer than the cap stays whole")

    headings = parser.add_argument_group("heading detection (ablation)")
    exclusive = headings.add_mutually_exclusive_group()
    exclusive.add_argument("--no-headings", action="store_true",
                           help="no heading boundaries at all — the reference arm")
    exclusive.add_argument("--line-headings", action="store_true",
                           help="line-based regex on the raw text, no LLM call; "
                                "isolates what the stronger regex contributes alone")
    exclusive.add_argument("--llm-headings", action="store_true",
                           help="line-based regex plus an LLM check where it finds "
                                "nothing (default without either flag: the shipped "
                                "sentence-based regex)")

    ablations = parser.add_argument_group("other ablations")
    ablations.add_argument("--no-filter", action="store_true",
                           help="keep low-information chunks; separates the boundary "
                                "effect from the filtering effect")
    ablations.add_argument("--midpoint-split", action="store_true",
                           help="at the size cap, cut in the middle instead of asking "
                                "the LLM for the best split point")
    ablations.add_argument("--enrich", action="store_true",
                           help="prefix every chunk with an LLM-generated topic line")
    return parser


def _heading_mode(args: argparse.Namespace) -> str:
    if args.llm_headings:
        return "hybrid"
    if args.line_headings:
        return "lines"
    return "regex"


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    Pipeline(
        document=args.document,
        chunker=ChunkerConfig(
            mode="window" if args.window else "incremental",
            step_sentences=args.step_sentences,
            max_chunk_sentences=args.max_chunk_sentences,
            max_chunk_chars=args.max_chunk_chars,
            respect_headings=not args.no_headings,
            heading_mode=_heading_mode(args),
            smart_split=not args.midpoint_split,
            filter_low_info=not args.no_filter,
            enrich=args.enrich,
            verbose=True,
        ),
        evaluation=EvalConfig(
            questions_file=args.questions_file,
            max_questions=args.max_questions,
            chunk_only=args.chunk_only,
            rechunk=args.rechunk,
        ),
    ).run()


if __name__ == "__main__":
    main()
