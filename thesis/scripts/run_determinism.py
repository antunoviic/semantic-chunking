from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parent.parent.parent))   # make the project root importable

from app.document_reader import DocumentReader
from llm_semantic_chunker import LLMChunker, OllamaClient

OUT = Path("eval_results/determinism.json")


def digest(chunks: list[str]) -> str:
    return hashlib.sha256("␞".join(chunks).encode("utf-8")).hexdigest()[:16]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("document")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--max-chunk-chars", type=int, default=1200)
    ap.add_argument("--max-chunk-sentences", type=int, default=100)
    ap.add_argument("--step-sentences", type=int, default=2)
    ap.add_argument("--no-headings", action="store_true")
    ap.add_argument("--no-filter", action="store_true")
    args = ap.parse_args()

    text = DocumentReader(args.document).extract_text()
    stem = Path(args.document).stem
    runs = []

    for i in range(1, args.runs + 1):
        print(f"\n=== run {i}/{args.runs} ===", flush=True)
        t0 = time.time()
        chunker = LLMChunker(
            client=OllamaClient(),
            mode="incremental",
            step_sentences=args.step_sentences,
            max_chunk_sentences=args.max_chunk_sentences,
            max_chunk_chars=args.max_chunk_chars,
            respect_headings=not args.no_headings,
            filter_low_info=not args.no_filter,
            verbose=False,
        )
        chunks = chunker.chunk(text)
        lens = [len(c) for c in chunks]
        runs.append({
            "run": i,
            "n_chunks": len(chunks),
            "avg_len": round(sum(lens) / max(1, len(lens))),
            "max_len": max(lens) if lens else 0,
            "digest": digest(chunks),
            "boundary_stats": chunker.boundary_stats,
            "seconds": round(time.time() - t0),
        })
        r = runs[-1]
        print(f"  {r['n_chunks']} Chunks | Ø {r['avg_len']} | digest {r['digest']} "
              f"| {r['seconds']} s | boundaries {r['boundary_stats']}")

    digests = {r["digest"] for r in runs}
    counts = [r["n_chunks"] for r in runs]
    identical = len(digests) == 1

    print("\n" + "=" * 66)
    if identical:
        print(f"  IDENTICAL across {args.runs} runs — digest {digests.pop()}")
        print("  -> averaging is unnecessary; report as 'determinism verified'.")
    else:
        mean = sum(counts) / len(counts)
        var = sum((c - mean) ** 2 for c in counts) / len(counts)
        print(f"  NOT identical: {len(digests)} different results across {args.runs} Laeufen")
        print(f"  chunk counts: {counts}  |  mean {mean:.1f}  |  sd {var ** 0.5:.1f}")
        print("  -> report the spread; fixed-size splitting has sd 0 for comparison.")
    print("=" * 66)

    OUT.parent.mkdir(exist_ok=True)
    prev = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
    prev[f"{stem}{'_noheadings' if args.no_headings else ''}"] = {
        "identical": identical, "runs": runs,
    }
    OUT.write_text(json.dumps(prev, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Written: {OUT}")


if __name__ == "__main__":
    main()
