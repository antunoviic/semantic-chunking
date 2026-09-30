from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from app.document_reader import DocumentReader
from eval.strategy_evaluator import build_strategies
from llm_semantic_chunker.incremental.heading_detection import extract_headings

CACHE = Path("chunks_cache")
MIN_TITLE_CHARS = 10  
START_TOLERANCE = 15     

# (name of variant, Cache-variant)
LLM_VARIANTS = [
    ("llm_incremental (no heading logic)", "incremental"),
    ("llm_incremental_headings (Regex, alt)", "incremental_headings"),
    ("llm_incremental_headings_hybrid (neu)", "incremental_headings_hybrid"),
]
BASELINES = ("fixed_512", "recursive")


def norm(s: str) -> str:
    return " ".join(s.split())


def reference_headings(raw: str) -> list[str]:
    """Unambiguously identifiable headings, longest first."""
    titles = {norm(t) for _, t in extract_headings(raw)}
    keep = {t for t in titles if len(t) >= MIN_TITLE_CHARS and t[0].isdigit()}
    return sorted(keep, key=len, reverse=True)


def spanning_chunks(chunks: list[str], heads: list[str]) -> int:
    """How many chunks contain a heading somewhere other than at their start?"""
    count = 0
    for c in chunks:
        cn = norm(c)
        if any(cn.find(h) > START_TOLERANCE for h in heads):
            count += 1
    return count


def load_cache(stem: str, variant: str) -> list[str] | None:
    path = CACHE / f"{stem}_{variant}.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["chunks"] if isinstance(data, dict) else data


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("document")
    ap.add_argument("--markdown", action="store_true",
                    help="print the table as Markdown, for pasting into reports")
    args = ap.parse_args()

    path = Path(args.document).expanduser()
    stem = path.stem
    raw = (path.read_text(encoding="utf-8") if path.suffix == ".txt"
           else DocumentReader(str(path)).extract_text())

    heads = reference_headings(raw)
    if not heads:
        print(f"No numbered headings found in {stem} — the measure "
              f"is not defined on this document.")
        return
    print(f"Document .............. {stem}")
    print(f"Reference ............. {len(heads)} unambiguously numbered headings\n")

    rows: list[tuple[str, int, int]] = []

    for label, variant in LLM_VARIANTS:
        chunks = load_cache(stem, variant)
        if chunks:
            rows.append((label, len(chunks), spanning_chunks(chunks, heads)))

    text = DocumentReader(str(path)).extract_text()
    base = build_strategies(text)
    for name in BASELINES:
        if name in base:
            rows.append((name, len(base[name]), spanning_chunks(base[name], heads)))

    rows.sort(key=lambda r: r[2] / max(1, r[1]), reverse=True)

    if args.markdown:
        print("| Variant | Chunks | heading inside | Share |")
        print("|---|---:|---:|---:|")
        for label, n, sp in rows:
            print(f"| {label} | {n} | {sp} | {sp/n*100:.1f} % |")
    else:
        print(f"  {'Variant':<40}{'Chunks':>8}{'spanning':>14}{'share':>9}")
        print("  " + "-" * 71)
        for label, n, sp in rows:
            print(f"  {label:<40}{n:>8}{sp:>14}{sp/n*100:>8.1f}%")


if __name__ == "__main__":
    main()
