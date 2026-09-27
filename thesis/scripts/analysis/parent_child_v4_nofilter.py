"""Parent-child retrieval on the v4 question sets, exact search.

Same construction as the harness (eval/strategy_evaluator.build_parent_child: children of
250 characters, overlap 30, LangChain's recursive splitter; eval/vectorstore.query with
dedupe_by_text: the top 6*k children are mapped to their parents, duplicates merged, the
first k parents returned, k = 10). Applied to the tested LLM arm and all five baselines.
Hit test and statistics are those of verify_independent.py (loaded with runpy).

Embeddings: children already embedded in chroma_db_eval (old run, same texts) are reused
after a spot check; the rest are embedded with bge-m3 via Ollama. Everything is cached in
eval_results/analysis/pc_embeddings.pkl.
Usage (repo root): python thesis/scripts/analysis/parent_child_v4_nofilter.py
"""
import contextlib, hashlib, io, json, math, pickle, random, runpy, sys, urllib.request
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
with contextlib.redirect_stdout(io.StringIO()):
    g = runpy.run_path(str(HERE / "verify_independent.py"))
qas, chunks, covered = g["qas"], g["chunks"], g["covered"]
binomtest, multipletests, cache0 = g["binomtest"], g["multipletests"], g["cache"]
from eval.strategy_evaluator import build_parent_child      # noqa: E402  (harness code)

ARMS = ["llm_incremental_nofilter", "packing_matched", "recursive_sent_matched", "semantic_matched",
        "recursive_matched", "fixed_matched"]
TARGET = {"nasa": 714, "rfc9110": 720, "wells": 725}
OUT = Path("eval_results/analysis"); OUT.mkdir(parents=True, exist_ok=True)
CACHE = OUT / "pc_embeddings.pkl"
key = lambda t: hashlib.sha1(t.encode("utf-8")).hexdigest()
emb = pickle.loads(CACHE.read_bytes()) if CACHE.exists() else {}


def ollama(texts):
    req = urllib.request.Request("http://localhost:11434/api/embed", method="POST",
                                 data=json.dumps({"model": "bge-m3", "input": [t[:5000] or " " for t in texts]}).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=600))["embeddings"]


# 1. reuse children embedded by the old run (Chroma), then spot-check them
if not emb and Path("chroma_db_eval").exists():
    import chromadb
    client = chromadb.PersistentClient(path="chroma_db_eval")
    names = [f"eval_{b}_{t}_parentchild" for t in TARGET.values() for b in ("fixed_matched", "recursive_matched")]
    names.append("eval_llm_incremental_parentchild")
    for n in names:
        got = client.get_collection(n).get(include=["embeddings", "documents"])
        for d, e in zip(got["documents"], got["embeddings"]):
            emb[key(d)] = np.asarray(e, dtype=np.float32)
    sample = random.Random(0).sample(sorted(emb), 30)
    rev = {}
    for n in names:
        for d in client.get_collection(n).get(include=["documents"])["documents"]:
            rev.setdefault(key(d), d)
    texts = [rev[k] for k in sample]
    fresh = np.array(ollama(texts), dtype=np.float64)
    old = np.vstack([emb[k] for k in sample]).astype(np.float64)
    cos = (fresh / np.linalg.norm(fresh, axis=1, keepdims=True) * old / np.linalg.norm(old, axis=1, keepdims=True)).sum(1)
    print(f"reused {len(emb)} child embeddings from chroma_db_eval; spot check cos min {cos.min():.6f}", flush=True)
    assert cos.min() > 0.999
    CACHE.write_bytes(pickle.dumps(emb))


def vecs(texts):
    need = [t for t in dict.fromkeys(texts) if key(t) not in emb]
    for i in range(0, len(need), 50):
        for t, v in zip(need[i:i + 50], ollama(need[i:i + 50])):
            emb[key(t)] = np.asarray(v, dtype=np.float32)
        if (i // 50) % 20 == 0 or i + 50 >= len(need):
            CACHE.write_bytes(pickle.dumps(emb))
            print(f"    embedded {min(i + 50, len(need))}/{len(need)}", flush=True)
    m = np.vstack([emb[key(t)] for t in texts]).astype(np.float64)
    return m / np.linalg.norm(m, axis=1, keepdims=True)


def pc_ranks(parents, qa, k_query=10, overfetch=6):
    children, parent_of = build_parent_child(parents)
    C = vecs(children)
    Qm = np.vstack([cache0[key(p["question"])] for p in qa]).astype(np.float64)
    Qm /= np.linalg.norm(Qm, axis=1, keepdims=True)
    order = np.argsort(-(Qm @ C.T), axis=1, kind="stable")[:, :k_query * overfetch]
    out = []
    for row, pair in zip(order, qa):
        seen, shown = set(), []
        for i in row:
            t = parent_of[i]
            if t not in seen:
                seen.add(t); shown.append(t)
            if len(shown) == k_query:
                break
        out.append(next((j for j, t in enumerate(shown[:3], 1) if covered(pair["source_text"], t)), None))
    return out, len(children)


R, RPC, n_children = g["R"], {}, {}
for doc in ("nasa", "rfc9110", "wells"):
    RPC[doc] = {}
    for arm in ARMS:
        print(f"  {doc} {arm}", flush=True)
        RPC[doc][arm], n_children[(doc, arm)] = pc_ranks(chunks[doc][arm], qas[doc])
CACHE.write_bytes(pickle.dumps(emb))

hit = lambda r, k: [(x or 99) <= k for x in r]
res = {"arms": {}, "tests": [], "gain": []}
print("\nHit rates with parent-child (without in parentheses)")
for doc in RPC:
    n = len(qas[doc]); res["arms"][doc] = {}
    for arm in ARMS:
        h = {k: sum(hit(RPC[doc][arm], k)) * 100 / n for k in (1, 3)}
        h0 = {k: sum(hit(R[doc][arm], k)) * 100 / n for k in (1, 3)}
        res["arms"][doc][arm] = dict(h1=h[1], h3=h[3], h1_plain=h0[1], h3_plain=h0[3], children=n_children[(doc, arm)])
        for k in (1, 3):
            a, b = hit(RPC[doc][arm], k), hit(R[doc][arm], k)
            x = sum(p and not q for p, q in zip(a, b)); y = sum(q and not p for p, q in zip(a, b))
            res["gain"].append(dict(doc=doc, arm=arm, k=k, x=x, y=y, d=(x - y) * 100 / n,
                                    p=binomtest(x, x + y, 0.5).pvalue if x + y else 1.0))
        print(f"  {doc:8} {arm:24} children {n_children[(doc, arm)]:5}  Hit@1 {h[1]:5.1f} ({h0[1]:5.1f})  Hit@3 {h[3]:5.1f} ({h0[3]:5.1f})")
    for base in ARMS[1:]:
        for k in (1, 3):
            a, b = hit(RPC[doc]["llm_incremental_nofilter"], k), hit(RPC[doc][base], k)
            x = sum(p and not q for p, q in zip(a, b)); y = sum(q and not p for p, q in zip(a, b))
            d = (x - y) / n; se = math.sqrt((x + y) / n - d * d) / math.sqrt(n)
            res["tests"].append(dict(doc=doc, base=base, k=k, x=x, y=y, d=d * 100, lo=(d - 1.96 * se) * 100,
                                     hi=(d + 1.96 * se) * 100, p=binomtest(x, x + y, 0.5).pvalue if x + y else 1.0))
rej, ph, _, _ = multipletests([t["p"] for t in res["tests"]], alpha=0.05, method="holm")
for t, r, p in zip(res["tests"], rej, ph):
    t["p_holm"], t["reject"] = float(p), bool(r)
print("\nLLM+PC vs baseline+PC, Holm over 30")
for t in res["tests"]:
    print(f"  {t['doc']:8} {t['base']:24} Hit@{t['k']} {t['d']:+5.1f} pp [{t['lo']:+5.1f}, {t['hi']:+5.1f}] {t['x']:3}:{t['y']:<3} p={t['p']:.4f} Holm={t['p_holm']:.4f}{' *' if t['reject'] else ''}")
print("\nGain from parent-child per arm (unadjusted)")
for t in res["gain"]:
    print(f"  {t['doc']:8} {t['arm']:24} Hit@{t['k']} {t['d']:+5.1f} pp  {t['x']:3}:{t['y']:<3} p={t['p']:.4f}")
json.dump(res, open(OUT / "parent_child_v4_nofilter.json", "w"), indent=1)
