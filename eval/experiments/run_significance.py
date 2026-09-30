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
from eval.vectorstore import OllamaEmbeddingFunction

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



# --------------------------------------------------------------------------
# Fast path: the evaluation already stored a rank per question per strategy in
# eval_results/<doc>_<timestamp>.json. Re-embedding to recover them costs about
# an hour per document and cannot change them, so --from-results reads them
# instead. This is also the only mode that applies the Holm correction, because
# the correction is defined over the whole family of tests (3 documents x 2
# metrics), not over one document at a time.
# --------------------------------------------------------------------------

DOCS = ("rfc9110", "wells", "nasa")
METRICS = (1, 3)
ALPHA = 0.05

# The evaluation files of run v4, named rather than found. Picking the newest
# file per document let any later exploratory run replace the reported input
# without notice. Override with --results doc=path.
PINNED_RESULTS = {
    "nasa":    "nasa_20260922_195008.json",
    "rfc9110": "rfc9110_20260922_181953.json",
    "wells":   "wells_20260922_185730.json",
}


def _pinned_results(doc: str, results_dir: Path, overrides: dict[str, Path]) -> Path | None:
    path = overrides.get(doc) or results_dir / PINNED_RESULTS[doc]
    if not path.exists():
        raise SystemExit(f"{doc}: pinned result file {path} does not exist")
    return path


def _paired_counts(A: list[dict], B: list[dict], k: int) -> tuple[int, int, int, int]:
    """Discordant counts b, c plus both hit rates, paired by position.

    Paired by index, not by question text: two questions in the rfc9110 set
    share their wording while pointing at different sections, and keying on the
    text would silently merge them.
    """
    ha = [bool(q["rank"]) and q["rank"] <= k for q in A]
    hb = [bool(q["rank"]) and q["rank"] <= k for q in B]
    b = sum(1 for x, y in zip(ha, hb) if x and not y)
    c = sum(1 for x, y in zip(ha, hb) if y and not x)
    return b, c, sum(ha), sum(hb)


def from_results(results_dir: Path, out: Path, overrides: dict[str, Path] | None = None) -> None:
    import math

    tests = []
    for doc in DOCS:
        path = _pinned_results(doc, results_dir, overrides or {})
        if path is None:
            print(f"  [skip] no results for {doc} in {results_dir}")
            continue
        rows = {r["strategy"]: r for r in json.loads(path.read_text(encoding="utf-8"))}
        if REFERENCE not in rows:
            print(f"  [skip] {doc}: no {REFERENCE} arm")
            continue
        baseline = next((s for s in rows if s.startswith("recursive_matched")
                         and not s.endswith("parentchild")), None)
        if baseline is None:
            print(f"  [skip] {doc}: no length-matched recursive baseline")
            continue
        A, B = rows[REFERENCE]["question_ranks"], rows[baseline]["question_ranks"]
        if len(A) != len(B) or any(a["question"] != b["question"] for a, b in zip(A, B)):
            raise SystemExit(f"{doc}: the two arms did not see the same question list")
        n = len(A)
        for k in METRICS:
            b, c, na, nb = _paired_counts(A, B, k)
            d = (b - c) / n
            se = math.sqrt(b + c - (b - c) ** 2 / n) / n
            tests.append(dict(doc=doc, baseline=baseline, k=k, n=n, b=b, c=c,
                              hit_a=na / n * 100, hit_b=nb / n * 100, delta=d * 100,
                              lo=(d - 1.96 * se) * 100, hi=(d + 1.96 * se) * 100,
                              p=mcnemar_p(b, c), source=path.name))

    if not tests:
        raise SystemExit("no tests could be formed")

    # Holm-Bonferroni: sort ascending, adjusted p is a running maximum of
    # (m - i) * p, and the chain stops at the first hypothesis not rejected.
    m = len(tests)
    tests.sort(key=lambda t: t["p"])
    running = 0.0
    for i, t in enumerate(tests):
        running = max(running, (m - i) * t["p"])
        t["p_adj"] = min(1.0, running)
    stopped = False
    for t in tests:
        t["reject"] = (not stopped) and t["p_adj"] <= ALPHA
        if not t["reject"]:
            stopped = True

    _write_holm_report(tests, out, m)


def _write_holm_report(tests: list[dict], out: Path, m: int) -> None:
    from datetime import datetime
    L = [
        f"# Significance of the primary comparison",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')} from eval_results/",
        "",
        f"`{REFERENCE}` against the length-matched recursive baseline. McNemar's "
        f"exact two-sided test on the discordant questions, Holm-Bonferroni over "
        f"m={m} tests (documents x metrics), alpha={ALPHA}.",
        "",
        "Holm is a step-down procedure: it stops at the first hypothesis it cannot "
        "reject, so the tests below that point were never reached rather than "
        "tested and passed.",
        "",
        "| Document | k | n | LLM | Baseline | delta | 95% CI | b | c | p | p (Holm) | Decision |",
        "|---|---:|---:|---:|---:|---:|:--:|---:|---:|---:|---:|---|",
    ]
    for t in tests:
        L.append(
            f"| `{t['doc']}` | {t['k']} | {t['n']} | {t['hit_a']:.1f} % | {t['hit_b']:.1f} % "
            f"| {t['delta']:+.1f} pp | [{t['lo']:+.1f}, {t['hi']:+.1f}] | {t['b']} | {t['c']} "
            f"| {t['p']:.5f} | {t['p_adj']:.5f} | "
            f"{'**rejected**' if t['reject'] else 'not rejected'} |")

    rejected = [t for t in tests if t["reject"]]
    L += ["", "## Reading", ""]
    if rejected:
        for t in rejected:
            L.append(f"- `{t['doc']}` Hit@{t['k']}: {t['delta']:+.1f} pp, "
                     f"p = {t['p']:.5f}, Holm-adjusted {t['p_adj']:.5f} — significant.")
    else:
        L.append("- No comparison survives the correction.")
    first_fail = next((t for t in tests if not t["reject"]), None)
    if first_fail:
        L.append(f"- The procedure stopped at `{first_fail['doc']}` Hit@{first_fail['k']} "
                 f"(adjusted p = {first_fail['p_adj']:.5f}).")

    L += ["", "## Equivalence", "",
          "A comparison counts as level only when the whole 95 % interval lies "
          "inside +-5 pp (TEST_PROTOCOL section 1). A large p-value alone does not "
          "establish equivalence.", ""]
    for t in sorted(tests, key=lambda x: (x["doc"], x["k"])):
        inside = t["lo"] >= -5 and t["hi"] <= 5
        L.append(f"- `{t['doc']}` Hit@{t['k']}: [{t['lo']:+.1f}, {t['hi']:+.1f}] pp — "
                 f"{'level' if inside else '**not decidable**'}")

    L += ["", "## Provenance", ""]
    for name in sorted({t["source"] for t in tests}):
        L.append(f"- `eval_results/{name}`")

    out.parent.mkdir(exist_ok=True)
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"\nWritten: {out}")

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-results", action="store_true",
                    help="read question ranks from eval_results/ instead of "
                         "re-embedding, and apply the Holm correction over all "
                         "documents (the only mode that does)")
    ap.add_argument("--results-dir", type=Path, default=Path("eval_results"))
    ap.add_argument("--results", nargs="+", default=[], metavar="DOC=PATH",
                    help="use these result files instead of the pinned v4 ones, "
                         "e.g. nasa=eval_results/nasa_20261001_120000.json")
    ap.add_argument("--out", type=Path, default=Path("thesis/results/SIGNIFICANCE.md"))
    ap.add_argument("--document")
    ap.add_argument("--questions")
    ap.add_argument("--label")
    args = ap.parse_args()

    if args.from_results:
        overrides = {}
        for item in args.results:
            doc, sep, path = item.partition("=")
            if not sep or doc not in DOCS:
                ap.error(f"--results expects DOC=PATH with DOC in {DOCS}, got {item!r}")
            overrides[doc] = Path(path)
        from_results(args.results_dir, args.out, overrides)
        return
    if not (args.document and args.questions and args.label):
        ap.error("--document, --questions and --label are required "
                 "unless --from-results is given")

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
