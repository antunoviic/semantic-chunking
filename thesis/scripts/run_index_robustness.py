from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import chromadb

from app.chunk_cache import ChunkCache
from app.document_reader import DocumentReader
from eval.strategy_evaluator import (StrategyEvaluator, build_parent_child,
                                     build_strategies)
from llm_semantic_chunker.vectorstore import OllamaEmbeddingFunction

OUT_DIR = Path("thesis/results")     # one report per document, see write_report
DEFAULT_EF = 100          # Chroma's default, which produced every number so far
REPORT_KS = (1, 3, 5, 10)


def embed_all(fn, texts: list[str], batch: int = 50) -> list[list[float]]:
    out: list[list[float]] = []
    for i in range(0, len(texts), batch):
        out.extend(fn(texts[i:i + batch]))
        print(f"    embedded {min(i + batch, len(texts))}/{len(texts)}", flush=True)
    return out


def make_collection(client, name: str, texts: list[str], embeddings: list,
                    display: list[str] | None, search_ef: int):
    try:
        client.delete_collection(name)
    except Exception:
        pass
    col = client.create_collection(
        name=name,
        metadata={"hnsw:space": "cosine", "hnsw:search_ef": search_ef},
    )
    shown = display if display is not None else texts
    for i in range(0, len(texts), 100):
        sl = slice(i, i + 100)
        col.add(
            ids=[f"c{j}" for j in range(i, min(i + 100, len(texts)))],
            embeddings=embeddings[sl],
            documents=texts[sl],
            metadatas=[{"shown": s} for s in shown[sl]],
        )
    return col


def run_queries(col, q_embeddings: list, qa: list[dict], k: int,
                dedupe: bool, overfetch: int) -> dict:
    hits = {kk: 0 for kk in REPORT_KS}
    rr: list[float] = []
    distinct_after_dedupe: list[int] = []

    for qe, pair in zip(q_embeddings, qa):
        want = k * overfetch if dedupe else k
        res = col.query(query_embeddings=[qe], n_results=min(want, col.count()))
        texts, seen = [], set()
        for meta in res["metadatas"][0]:
            t = meta["shown"]
            if dedupe:
                if t in seen:
                    continue
                seen.add(t)
            texts.append(t)
            if len(texts) >= k:
                break
        distinct_after_dedupe.append(len(texts))

        rank = next((j for j, t in enumerate(texts, 1)
                     if StrategyEvaluator._is_hit(pair["source_text"], t)), None)
        if rank:
            for kk in REPORT_KS:
                if rank <= kk:
                    hits[kk] += 1
            rr.append(1.0 / rank)
        else:
            rr.append(0.0)

    n = len(qa)
    return {
        **{f"hit@{kk}": round(hits[kk] / n * 100, 1) for kk in REPORT_KS},
        "mrr": round(sum(rr) / n, 3),
        "min_returned": min(distinct_after_dedupe),
        "avg_returned": round(sum(distinct_after_dedupe) / n, 1),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--document", default="docs/rfc9110.txt")
    ap.add_argument("--questions", default="eval_cache/rfc9110_questions_v3.json")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--overfetch", type=int, default=6,
                    help="matches VectorStore._DEDUPE_OVERFETCH")
    args = ap.parse_args()

    qa = json.loads(Path(args.questions).read_text(encoding="utf-8"))
    text = DocumentReader(args.document).extract_text()
    stem = Path(args.document).stem

    sets: dict[str, tuple[list[str], list[str] | None]] = {}
    base = build_strategies(text)
    for name in ("fixed_256", "fixed_512", "recursive"):
        if name in base:
            sets[name] = (base[name], None)

    llm = ChunkCache().load(args.document, variant="incremental")
    if llm:
        sets["llm_incremental"] = (llm, None)
        children, parents = build_parent_child(llm)
        sets["parent_child"] = (children, parents)
    else:
        print(f"[warn] no chunk cache for {stem} — LLM strategies skipped")

    fn = OllamaEmbeddingFunction()
    print(f"\n[embed] {len(qa)} questions")
    q_emb = embed_all(fn, [p["question"] for p in qa])

    client = chromadb.EphemeralClient()      # nothing is written to disk
    rows: list[dict] = []

    for name, (chunks, display) in sets.items():
        print(f"\n[{name}] {len(chunks)} vectors — embedding")
        emb = embed_all(fn, chunks)
        dedupe = display is not None
        exhaustive = max(DEFAULT_EF + 1, len(chunks))    # >= collection size = exact

        for label, ef in (("default", DEFAULT_EF), ("near-exhaustive", exhaustive)):
            col = make_collection(client, f"{name}_{ef}", chunks, emb, display, ef)
            m = run_queries(col, q_emb, qa, args.k, dedupe, args.overfetch)
            rows.append({"strategy": name, "variant": label, "search_ef": ef,
                         "vectors": len(chunks), **m})
            print(f"  ef={ef:<6} Hit@1 {m['hit@1']:>5} %  MRR {m['mrr']:.3f}"
                  f"  (min. returned: {m['min_returned']})")

    write_report(rows, stem, args)


def write_report(rows: list[dict], stem: str, args) -> None:
    from datetime import datetime
    L = [
        "# Robustness against the approximate search index",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')} — document `{stem}`, "
        f"{Path(args.questions).name}, k={args.k}",
        "",
        "HNSW searches approximately. The default (`search_ef=100`, which produced "
        "every result so far) is compared against a near-exhaustive search "
        "(`search_ef` >= collection size). The difference is the error of the "
        "approximation, and therefore a **measured** rather than estimated noise floor.",
        "",
        "| Strategy | Vectors | Variant | Hit@1 | Hit@3 | Hit@10 | MRR |",
        "|---|---:|---|---:|---:|---:|---:|",
    ]
    for r in rows:
        L.append(f"| {r['strategy']} | {r['vectors']} | {r['variant']} "
                 f"| {r['hit@1']} % | {r['hit@3']} % | {r['hit@10']} % | {r['mrr']:.3f} |")
    L += ["", "## Difference per strategy", "",
          "| Strategy | Vectors | Δ Hit@1 | Δ MRR |", "|---|---:|---:|---:|"]
    for i in range(0, len(rows), 2):
        a, b = rows[i], rows[i + 1]
        L.append(f"| {a['strategy']} | {a['vectors']} | "
                 f"{b['hit@1'] - a['hit@1']:+.1f} pp | {b['mrr'] - a['mrr']:+.3f} |")

    pc = [r for r in rows if r["strategy"] == "parent_child"]
    if pc:
        L += ["", f"## Is `_DEDUPE_OVERFETCH = {args.overfetch}` enough?", "",
              f"k={args.k} results are requested. If fewer distinct parents remain "
              f"after de-duplication, Hit@{args.k} drops for purely technical "
              f"reasons.", "",
              "| Variant | minimum returned | mean |",
              "|---|---:|---:|"]
        for r in pc:
            L.append(f"| {r['variant']} | {r['min_returned']} | {r['avg_returned']} |")
        worst = min(r["min_returned"] for r in pc)
        if worst < args.k:
            verdict = (f"**Insufficient** — at least one question received only {worst} "
                       f"instead of {args.k} parents. Raise the overfetch factor.")
        else:
            verdict = (f"Sufficient: even in the worst case {worst} "
                       f"distinct parents came back.")
        L += ["", f"> {verdict}"]

    out = OUT_DIR / f"INDEX_ROBUSTNESS_{stem}.md"
    OUT_DIR.mkdir(exist_ok=True)
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"\nWritten: {out}")


if __name__ == "__main__":
    main()
