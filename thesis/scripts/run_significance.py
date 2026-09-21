from __future__ import annotations

import argparse
import json
import sys
from math import comb
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import chromadb

from app.chunk_cache import ChunkCache
from app.document_reader import DocumentReader
from eval.strategy_evaluator import (StrategyEvaluator, build_parent_child,
                                     build_strategies)
from llm_semantic_chunker.vectorstore import OllamaEmbeddingFunction

OUT = Path("thesis/results/SIGNIFICANCE.md")
FETCH_K = 10
REPORT_KS = (1, 3, 10)
DEDUPE_OVERFETCH = 6

# Every other strategy is tested against this reference
REFERENCE = "llm_incremental"


def mcnemar_p(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, sum(comb(n, i) for i in range(k + 1)) / 2 ** n * 2)


def embed_all(fn, texts: list[str], batch: int = 50) -> list:
    out = []
    for i in range(0, len(texts), batch):
        out.extend(fn(texts[i:i + batch]))
        if (i // batch) % 10 == 0:
            print(f"      {min(i + batch, len(texts))}/{len(texts)}", flush=True)
    return out


def ranks_for(client, fn, name, chunks, display, q_emb, qa) -> list[int | None]:
    """Rank of the first hitting result per question (None = outside the top 10)."""
    try:
        client.delete_collection(name)
    except Exception:
        pass
    col = client.create_collection(name=name, metadata={"hnsw:space": "cosine"})
    shown = display if display is not None else chunks
    emb = embed_all(fn, chunks)
    for i in range(0, len(chunks), 100):
        sl = slice(i, i + 100)
        col.add(ids=[f"c{j}" for j in range(i, min(i + 100, len(chunks)))],
                embeddings=emb[sl], documents=chunks[sl],
                metadatas=[{"shown": s} for s in shown[sl]])

    dedupe = display is not None
    want = FETCH_K * DEDUPE_OVERFETCH if dedupe else FETCH_K
    out: list[int | None] = []
    for qe, pair in zip(q_emb, qa):
        res = col.query(query_embeddings=[qe], n_results=min(want, col.count()))
        texts, seen = [], set()
        for meta in res["metadatas"][0]:
            t = meta["shown"]
            if dedupe:
                if t in seen:
                    continue
                seen.add(t)
            texts.append(t)
            if len(texts) >= FETCH_K:
                break
        out.append(next((j for j, t in enumerate(texts, 1)
                         if StrategyEvaluator._is_hit(pair["source_text"], t)), None))
    return out


def hit_at(rank: int | None, k: int) -> bool:
    return rank is not None and rank <= k


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--document", required=True)
    ap.add_argument("--questions", required=True)
    ap.add_argument("--label", required=True)
    args = ap.parse_args()

    doc = str(Path(args.document).expanduser())
    qa = json.loads(Path(args.questions).read_text(encoding="utf-8"))
    text = DocumentReader(doc).extract_text()
    cache = ChunkCache()

    llm = cache.load(doc, variant="incremental")
    if not llm:
        raise SystemExit(f"No chunk cache for {doc}")
    match_len = round(sum(map(len, llm)) / len(llm))
    base = build_strategies(text, match_len=match_len)

    sets: dict[str, tuple[list[str], list[str] | None]] = {REFERENCE: (llm, None)}
    hybrid = cache.load(doc, variant="incremental_headings_hybrid")
    if hybrid:
        sets["llm_incremental_headings_hybrid"] = (hybrid, None)
    children, parents = build_parent_child(llm)
    sets["llm_incremental_parentchild"] = (children, parents)
    for name in base:
        if name.startswith(("recursive_matched", "fixed_matched")) or name == "recursive":
            sets[name] = (base[name], None)

    fn = OllamaEmbeddingFunction()
    print(f"[{args.label}] {len(qa)} questions, {len(sets)} strategies")
    print("  embedding questions")
    q_emb = embed_all(fn, [p["question"] for p in qa])

    client = chromadb.EphemeralClient()
    ranks: dict[str, list] = {}
    for name, (chunks, display) in sets.items():
        print(f"  [{name}] {len(chunks)} vectors")
        ranks[name] = ranks_for(client, fn, name, chunks, display, q_emb, qa)
        for k in REPORT_KS:
            n_hit = sum(hit_at(r, k) for r in ranks[name])
            print(f"      Hit@{k}: {n_hit/len(qa)*100:.1f} %")

    write_report(args.label, len(qa), ranks)


def write_report(label: str, n: int, ranks: dict) -> None:
    from datetime import datetime
    L = [f"## {label}  (n = {n})", "",
         f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", "",
         f"Every strategy against `{REFERENCE}`, paired McNemar test on the "
         f"discordant questions.", ""]
    for k in REPORT_KS:
        L += [f"### Hit@{k}", "",
              f"| Strategy | Hit@{k} | this only | {REFERENCE} only | p | |",
              "|---|---:|---:|---:|---:|---|"]
        ref = ranks[REFERENCE]
        for name, rr in ranks.items():
            rate = sum(hit_at(r, k) for r in rr) / n * 100
            if name == REFERENCE:
                L.append(f"| **{name}** | {rate:.1f} % | — | — | — | reference |")
                continue
            b = sum(1 for x, y in zip(rr, ref) if hit_at(x, k) and not hit_at(y, k))
            c = sum(1 for x, y in zip(rr, ref) if not hit_at(x, k) and hit_at(y, k))
            p = mcnemar_p(b, c)
            if p >= 0.05:
                verdict = "not significant"
            else:
                verdict = "**besser**" if b > c else "**schlechter**"
            L.append(f"| {name} | {rate:.1f} % | {b} | {c} | {p:.4f} | {verdict} |")
        L.append("")

    OUT.parent.mkdir(exist_ok=True)
    prev = OUT.read_text(encoding="utf-8") if OUT.exists() else \
        ("# significance tests\n\n"
         "McNemar on the discordant questions, exact two-sided binomial test.\n"
         "Tested on **Hit@1** — the metric that is also reported. A test on Hit@10\n"
         "alone can point the other way; see rfc9110.\n\n---\n\n")
    marker = f"## {label}  (n ="
    if marker in prev:
        head, rest = prev.split(marker, 1)
        tail = rest.split("\n---\n", 1)
        prev = head + (tail[1] if len(tail) > 1 else "")
    OUT.write_text(prev.rstrip() + "\n\n---\n\n" + "\n".join(L) + "\n",
                   encoding="utf-8")
    print(f"\nGeschrieben: {OUT}")


if __name__ == "__main__":
    main()
