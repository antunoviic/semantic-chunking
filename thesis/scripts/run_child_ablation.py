"""Ablation over the CHILD size in parent-child retrieval.

Question: does the measured parent-child gain depend on the arbitrarily chosen
value of 250 characters, or is it an effect of the principle?

ONLY the child size is varied. Held constant:
  * the parents (the cached v4 reference arm, variant "incremental")
  * the text returned (always the parent)
  * question set, embedder, k, overlap (30 characters)

The row "baseline (parents only)" is the baseline: searching the parents
directly, i.e. the plain llm_incremental arm.

The parents come from the cache, so no chunking happens here and no model is
asked for a boundary: the children are derived arithmetically. Only embedding
and retrieval cost time. A parent cache that does not match the running code is
refused by ChunkCache.load, so the ablation cannot silently mix code states.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parent.parent.parent))   # make the project root importable

from app.chunk_cache import ChunkCache
from eval.strategy_evaluator import StrategyEvaluator, build_parent_child
from eval.vectorstore import VectorStore

CHILD_SIZES = [150, 250, 400]
OVERLAP = 30
K = 3

DOCS_DIR = Path(os.environ.get("DOCS_DIR", "docs"))

# (label, document, question set) — the three documents of run v4
RUNS = [
    ("rfc9110", DOCS_DIR / "rfc9110.txt", "eval_cache/rfc9110_questions_v3.json"),
    ("wells",   DOCS_DIR / "wells.txt",   "eval_cache/wells_questions.json"),
    ("nasa",    DOCS_DIR / "nasa.pdf",    "eval_cache/nasa_questions_literal_v3.json"),
]

OUT_JSON = Path("eval_results/child_ablation.json")
OUT_MD = Path("thesis/results/CHILD_ABLATION.md")


def load_parents(document: Path) -> list[str] | None:
    """The v4 reference arm, or None if it is missing or from other code."""
    if not document.exists():
        return None
    return ChunkCache().load(str(document), variant="incremental")


def main() -> None:
    store = VectorStore(persist_dir="./chroma_db_eval")
    ev = StrategyEvaluator(store, k=K)
    all_rows: dict[str, list[dict]] = {}

    for label, document, qfile in RUNS:
        parents = load_parents(document)
        if not parents:
            print(f"!! skipping {label} — no usable parent cache for {document}")
            continue
        if not Path(qfile).exists():
            print(f"!! skipping {label} — question set missing: {qfile}")
            continue
        qa = json.loads(Path(qfile).read_text(encoding="utf-8"))

        print(f"\n{'='*70}\n>>> {label}   {len(parents)} parents, {len(qa)} questions\n{'='*70}")
        rows = []

        # Baseline: search the parents directly (= the llm_incremental arm)
        col = f"probe_child_{label}_base"
        r = ev.evaluate(col, parents, qa)
        r["child_chars"] = None
        r["n_children"] = len(parents)
        rows.append(r)
        print(f"  baseline (parents only)  Hit@1 {r['hit_rate@1']:5.1f}%  MRR {r['mrr']:.3f}")

        for size in CHILD_SIZES:
            children, parent_of = build_parent_child(parents, child_chars=size,
                                                     child_overlap=OVERLAP)
            col = f"probe_child_{label}_{size}"
            r = ev.evaluate(col, children, qa, display_texts=parent_of)
            r["child_chars"] = size
            r["n_children"] = len(children)
            rows.append(r)
            print(f"  children {size:>3} chars       Hit@1 {r['hit_rate@1']:5.1f}%  "
                  f"MRR {r['mrr']:.3f}  ({len(children)} children)")

        all_rows[label] = rows

    # Clean up: do not leave the probe collections in the store
    for name in store.list_collections():
        if name.startswith("eval_probe_child_"):
            try:
                store.delete_collection(name)
            except Exception:
                pass

    OUT_JSON.parent.mkdir(exist_ok=True)
    OUT_JSON.write_text(json.dumps(all_rows, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Ablation: child size in parent-child retrieval",
        "",
        "**Only** the child size is varied. Parents, returned text, question set, "
        "embedder, k=3 and overlap (30 characters) are held constant.",
        "",
        "`Baseline` = searching the parents directly (equals the `llm_incremental` arm).",
        "",
    ]
    for label, rows in all_rows.items():
        base = rows[0]
        lines += [f"## {label}", "",
                  "| Child size | Children | Hit@1 | Hit@3 | MRR | Δ MRR vs. baseline | Ctx |",
                  "|---|---:|---:|---:|---:|---:|---:|"]
        for r in rows:
            name = "baseline (parents only)" if r["child_chars"] is None else f"{r['child_chars']} chars"
            delta = "—" if r["child_chars"] is None else f"{r['mrr']-base['mrr']:+.3f}"
            lines.append(
                f"| {name} | {r['n_children']} | {r['hit_rate@1']} % | "
                f"{r.get('hit_rate@3', 0)} % | {r['mrr']:.3f} | {delta} | {r['avg_retrieved_chars']} |")
        lines.append("")
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nWritten: {OUT_JSON} and {OUT_MD}")


if __name__ == "__main__":
    main()
