from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path as _Path

sys.path.insert(0, str(_Path(__file__).resolve().parent.parent.parent))   # make the project root importable

from app.document_reader import DocumentReader
from llm_semantic_chunker import OllamaClient
from llm_semantic_chunker.incremental import (HeadingOnlyPrompt, heading_sentence_indices,
                                     looks_like_heading_candidate)
from llm_semantic_chunker.text_splitter import TextSplitter

OUT = _Path("thesis/results/heading_detection_report.json")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("document")
    ap.add_argument("--limit", type=int, default=250,
                    help="max. model calls (sample of the candidates)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--window", type=int, default=2,
                    help="sentences per candidate (matches step_sentences in the detector)")
    args = ap.parse_args()

    stem = _Path(args.document).stem
    raw = (_Path(args.document).read_text(encoding="utf-8")
           if args.document.endswith(".txt")
           else DocumentReader(args.document).extract_text())
    text = DocumentReader(args.document).extract_text()
    sentences = TextSplitter().split_sentences(text)

    regex_idx = heading_sentence_indices(raw, sentences)

    def probe_at(i: int) -> str:
        return " ".join(sentences[i:i + args.window])[:120].strip()

    candidates = [i for i in range(len(sentences)) if looks_like_heading_candidate(probe_at(i))]

    print(f"Document .............. {stem}  ({len(sentences)} sentences)")
    print(f"Regex reference ....... {len(regex_idx)} headings")
    print(f"Structural filter ..... {len(candidates)} candidates "
          f"({len(candidates)*100//max(1,len(sentences))} % of sentences)")

    # every regex hit (for recall) plus a fill of remaining candidates
    random.seed(args.seed)
    must = [i for i in candidates if i in regex_idx]
    rest = [i for i in candidates if i not in regex_idx]
    random.shuffle(rest)
    sample = sorted(must + rest[:max(0, args.limit - len(must))])
    print(f"Sample ................ {len(sample)} model calls "
          f"({len(must)} regex hits + {len(sample)-len(must)} more)\n")

    client, prompt = OllamaClient(), HeadingOnlyPrompt()
    hits_on_regex, llm_only, missed = 0, [], []

    for n, idx in enumerate(sample, 1):
        if n % 25 == 0:
            print(f"  [{n}/{len(sample)}] ...", flush=True)
        verdict = prompt.parse(client.chat(prompt.as_messages(probe_at(idx))).strip())
        if idx in regex_idx:
            if verdict:
                hits_on_regex += 1
            else:
                missed.append(probe_at(idx)[:90])
        elif verdict:
            llm_only.append(probe_at(idx)[:90])

    n_ref = len(must) or 1
    recall = hits_on_regex / n_ref * 100

    print("\n" + "=" * 70)
    print(f"  RECALL against the regex reference : {hits_on_regex}/{len(must)} = {recall:.1f} %")
    print(f"  Extra findings by the model   : {len(llm_only)} of "
          f"{len(sample)-len(must)} non-reference candidates checked")
    print("=" * 70)

    if missed:
        print(f"\n  MISSED BY THE MODEL (real headings, {len(missed)}):")
        for s in missed[:12]:
            print(f"    {s!r}")
    if llm_only:
        print(f"\n  EXTRA FINDINGS — inspect whether these are real headings ({len(llm_only)}):")
        for s in llm_only[:20]:
            print(f"    {s!r}")

    OUT.parent.mkdir(exist_ok=True)
    prev = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
    prev[stem] = {
        "sentences": len(sentences), "regex_headings": len(regex_idx),
        "structural_candidates": len(candidates), "sampled": len(sample),
        "recall_on_regex_pct": round(recall, 1),
        "missed_by_llm": missed, "llm_only_findings": llm_only,
    }
    OUT.write_text(json.dumps(prev, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nGeschrieben: {OUT}")


if __name__ == "__main__":
    main()
