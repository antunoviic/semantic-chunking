"""Live demo: evaluate the chunks that were produced beforehand and print the result table.

The expensive part, chunking with qwen3.5:4b (2 to 6 hours per document), was run
beforehand and cached in chunks_cache/. This script only does the evaluation, from
the cached chunks and the cached bge-m3 embeddings: exact cosine search, the 80 %
hit criterion, McNemar and Holm over all 30 comparisons. No model call, no network.

Run from the repository root:
    PYTHONPATH=. python eval/experiments/analysis/demo_evaluation.py             # all three documents
    PYTHONPATH=. python eval/experiments/analysis/demo_evaluation.py rfc9110     # one document in detail
"""
import hashlib, json, math, os, pickle, re, sys, time
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.stats import binomtest
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, str(Path.cwd()))
from app.document_reader import DocumentReader          # noqa: E402
from eval.strategy_evaluator import build_strategies     # noqa: E402

DOCS = {"rfc9110": ("RFC 9110", "docs/rfc9110.txt", "eval_cache/rfc9110_questions_v4.json"),
        "nasa": ("NASA SE Handbook", "docs/nasa.pdf", "eval_cache/nasa_questions_literal_v4.json"),
        "wells": ("Wells, history", "docs/wells.txt", "eval_cache/wells_questions.json")}
ARMS = [("llm_incremental_nofilter", "LLM chunker (this work)"),
        ("recursive_sent_matched", "Recursive, sentence ends first"),
        ("packing_matched", "Same chunker, model taken out"),
        ("semantic_matched", "Semantic (embeddings), tuned"),
        ("recursive_matched", "Recursive, LangChain default"),
        ("fixed_matched", "Fixed-size")]
REF = ARMS[0][0]

COLOR = sys.stdout.isatty() and not os.environ.get("NO_COLOR")
def c(s, code): return f"\033[{code}m{s}\033[0m" if COLOR else s
bold, dim, green, orange = (lambda s: c(s, "1")), (lambda s: c(s, "2")), (lambda s: c(s, "32")), (lambda s: c(s, "33"))
norm = lambda s: " ".join(s.split())


def fmt_p(p): return "<0.001" if p < 0.001 else f"{p:.3f}"


def step(n, text): print(f"\n{bold(orange(f'[{n}]'))} {bold(text)}")


def load_embeddings():
    return pickle.loads(Path("eval_results/fair_baselines_embeddings.pkl").read_bytes())


def vecs(cache, texts):
    m = np.vstack([cache[hashlib.sha1(t.encode("utf-8")).hexdigest()] for t in texts]).astype(np.float64)
    return m / np.linalg.norm(m, axis=1, keepdims=True)


def covered(anchor, chunk):
    """Hit criterion: at least 80 % of the anchor, in one piece, inside the chunk."""
    a, b = norm(anchor), norm(chunk)
    m = -(-len(a) * 4 // 5)
    return any(a[i:i + m] in b for i in range(len(a) - m + 1))


def first_hit(cache, chunks, Q, qa, kmax=3):
    order = np.argsort(-(Q @ vecs(cache, chunks).T), axis=1, kind="stable")[:, :kmax]
    return [next((j for j, i in enumerate(row, 1) if covered(p["source_text"], chunks[i])), None)
            for row, p in zip(order, qa)]


FIXED_CACHE = Path("eval_results/analysis/demo_fixed_matched_chunks.json")


def fixed_chunks(doc, path, target):
    """The fixed-size baseline is rebuilt from the document once and then cached (the PDF takes ~15 s)."""
    cached = json.loads(FIXED_CACHE.read_text()) if FIXED_CACHE.exists() else {}
    FIXED_CACHE.parent.mkdir(parents=True, exist_ok=True)
    if doc not in cached:
        cached[doc] = build_strategies(DocumentReader(path).extract_text(), match_len=target)[f"fixed_matched_{target}"]
        FIXED_CACHE.write_text(json.dumps(cached))
    return cached[doc]


def main():
    only = sys.argv[1] if len(sys.argv) > 1 else None
    t0 = time.time()
    v4 = json.load(open("eval_results/fair_baselines_v4.json"))["results"]
    print(bold("LLM-based semantic chunking · evaluation of the pre-computed chunks"))

    step(1, "Chunks produced beforehand by the LLM chunker (qwen3.5:4b via Ollama, cached)")
    chunks, qas = {}, {}
    for doc, (name, path, qf) in DOCS.items():
        cc = json.load(open(f"chunks_cache/{Path(path).stem}_incremental_nofilter.json"))
        bs = cc["params"]["boundary_stats"]; n = len(cc["chunks"])
        when = datetime.fromisoformat(cc["chunked_at"]).strftime("%d.%m. %H:%M")
        info = f"chunked {when}, code {cc['code_digest']}"
        print(f"  {name:17} {n:5} chunks   topic decisions {bs['semantic']:4}   size cap {bs['size_cap']:4}   {dim(info)}")
        target = v4[doc]["target"]
        chunks[doc] = {a: v4[doc]["chunks"][a] for a, _ in ARMS if a in v4[doc]["chunks"]}
        assert chunks[doc][REF] == cc["chunks"]
        chunks[doc]["fixed_matched"] = fixed_chunks(doc, path, target)
        qas[doc] = json.load(open(qf))

    step(2, "Baselines from LangChain, tuned to the same chunk length (714 / 720 / 725 characters)")
    print("  recursive (default and with sentence ends), semantic chunker, fixed-size, and the same chunker without the model")

    step(3, f"{sum(map(len, qas.values()))} questions · bge-m3 embeddings (cached) · exact cosine search")
    print("  hit = at least 80 % of the answer passage, in one piece, among the top k chunks")
    cache = load_embeddings()
    R = {}
    for doc in DOCS:
        Q = vecs(cache, [p["question"] for p in qas[doc]])
        R[doc] = {a: first_hit(cache, chunks[doc][a], Q, qas[doc]) for a, _ in ARMS}

    hit = lambda ranks, k: [r is not None and r <= k for r in ranks]
    tests = []
    for doc in DOCS:
        n = len(qas[doc])
        for a, _ in ARMS[1:]:
            for k in (1, 3):
                x, y = hit(R[doc][REF], k), hit(R[doc][a], k)
                b = sum(p and not q for p, q in zip(x, y)); cc_ = sum(q and not p for p, q in zip(x, y))
                tests.append(dict(doc=doc, arm=a, k=k, d=(b - cc_) * 100 / n, p=binomtest(b, b + cc_, 0.5).pvalue if b + cc_ else 1.0))
    rej, padj, _, _ = multipletests([t["p"] for t in tests], alpha=0.05, method="holm")
    for t, r, p in zip(tests, rej, padj): t["sig"], t["padj"] = bool(r), p
    sig = {(t["doc"], t["arm"], t["k"]): t for t in tests}

    step(4, "Hit@k in percent, all splitters at the same chunk length")
    docs = [only] if only else list(DOCS)
    head = f"  {'':32}" + "".join(f"{DOCS[d][0]:>18}" for d in docs)
    sub = f"  {'':32}" + "".join(f"{'Hit@1':>10}{'Hit@3':>8}" for d in docs)
    print(bold(head)); print(dim(sub))
    for a, label in ARMS:
        cells = ""
        for d in docs:
            for k, w in ((1, 10), (3, 8)):
                v = f"{sum(hit(R[d][a], k)) * 100 / len(qas[d]):.1f}"
                mark = ""
                if a != REF and sig[(d, a, k)]["sig"]:
                    mark = "*"
                cell = f"{v}{mark}".rjust(w)
                cells += green(cell) if mark else cell
        line = f"  {label:32}{cells}"
        print(bold(line) if a == REF else line)
    print(dim("  * LLM chunker significantly better (McNemar, Holm-adjusted p < 0.05 over 30 comparisons)"))

    step(5, "Paired tests against the LLM chunker")
    n_sig = sum(t["sig"] for t in tests)
    count = f"{n_sig} of {len(tests)}"
    print(f"  {bold(count)} comparisons significant after the Holm correction:")
    for t in sorted((t for t in tests if t["sig"]), key=lambda t: t["padj"]):
        print(f"    {DOCS[t['doc']][0]:17} vs {dict(ARMS)[t['arm']]:30} Hit@{t['k']}  {t['d']:+5.1f} pp   p_adj {fmt_p(t['padj'])}")
    near = [t for t in tests if not t["sig"] and t["padj"] < 0.1]
    for t in near:
        print(dim(f"    nearest miss: {DOCS[t['doc']][0]} vs {dict(ARMS)[t['arm']]} Hit@{t['k']}  {t['d']:+.1f} pp   p_adj {t['padj']:.3f}"))
    print(f"  none against a splitter that keeps sentences intact")
    print(dim(f"\n  evaluation took {time.time() - t0:.0f} s; the chunking before it took 2 to 6 hours per document"))


if __name__ == "__main__":
    main()
