"""Exploratory: LLM-arm ablations (one setting changed, same code digest 3c3df72cca46)
evaluated on the v4 question sets with the independent functions of verify_independent.py.
New embeddings are cached in eval_results/analysis/ablation_embeddings.pkl."""
import hashlib, json, math, pickle, sys, urllib.request, runpy, io, contextlib
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
S = Path("eval_results/analysis"); S.mkdir(parents=True, exist_ok=True); EXTRA = S / "ablation_embeddings.pkl"
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    g = runpy.run_path(str(HERE / "verify_independent.py"))
cache, qas, covered, norm, TOPIC, binomtest = g["cache"], g["qas"], g["covered"], g["norm"], g["TOPIC"], g["binomtest"]
extra = pickle.loads(EXTRA.read_bytes()) if EXTRA.exists() else {}
key = lambda t: hashlib.sha1(t.encode("utf-8")).hexdigest()

def embed(texts):
    need = [t for t in dict.fromkeys(texts) if key(t) not in cache and key(t) not in extra]
    for i in range(0, len(need), 50):
        batch = need[i:i + 50]
        req = urllib.request.Request("http://localhost:11434/api/embed", method="POST",
              data=json.dumps({"model": "bge-m3", "input": [t[:5000] or " " for t in batch]}).encode(),
              headers={"Content-Type": "application/json"})
        for t, v in zip(batch, json.load(urllib.request.urlopen(req, timeout=600))["embeddings"]):
            extra[key(t)] = np.asarray(v, dtype=np.float32)
        EXTRA.write_bytes(pickle.dumps(extra)); print(f"    embedded {min(i+50, len(need))}/{len(need)}", flush=True)
    m = np.vstack([cache.get(key(t), extra.get(key(t))) for t in texts]).astype(np.float64)
    return m / np.linalg.norm(m, axis=1, keepdims=True)

def ranks(chunks, qa):
    Q = embed([p["question"] for p in qa]); S_ = Q @ embed(chunks).T
    order = np.argsort(-S_, axis=1, kind="stable")[:, :3]
    return [next((j for j, i in enumerate(row, 1) if covered(p["source_text"], chunks[i])), None) for row, p in zip(order, qa)]

ARMS = {"nasa": ["incremental", "incremental_midpoint", "incremental_headings_lines", "incremental_headings_hybrid", "incremental_enriched"],
        "rfc9110": ["incremental", "incremental_headings_lines", "incremental_headings_hybrid", "incremental_enriched"],
        "wells": ["incremental", "incremental_headings_lines", "incremental_headings_hybrid", "incremental_enriched"]}
out = {}
for doc, arms in ARMS.items():
    qa = qas[doc]; n = len(qa); R = {}
    for a in arms:
        d = json.load(open(f"chunks_cache/{doc}_{a}.json"))
        assert d["code_digest"].startswith("3c3df72cca46"), (doc, a, d["code_digest"])
        print(f"  {doc} {a}: {len(d['chunks'])} chunks", flush=True)
        R[a] = ranks(d["chunks"], qa)
        ch = d["chunks"]; out.setdefault(doc, {})[a] = dict(n_chunks=len(ch), mean=round(sum(map(len, ch)) / len(ch)))
    for a in arms:
        row = out[doc][a]
        for k in (1, 3):
            h = [(x or 99) <= k for x in R[a]]; ref = [(x or 99) <= k for x in R["incremental"]]
            x = sum(p and not q for p, q in zip(h, ref)); y = sum(q and not p for p, q in zip(h, ref))
            row[f"h{k}"] = sum(h) * 100 / n
            row[f"d{k}"] = (x - y) * 100 / n; row[f"b{k}"] = (x, y)
            row[f"p{k}"] = binomtest(x, x + y, 0.5).pvalue if x + y else 1.0
json.dump(out, open(S / "ablations_v4.json", "w"), indent=1)
for doc, rows in out.items():
    print(f"\n{doc} (n={len(qas[doc])})")
    for a, r in rows.items():
        print(f"  {a:30} {r['n_chunks']:5} chunks mean {r['mean']:4}  Hit@1 {r['h1']:5.1f} ({r['d1']:+5.1f}, {r['b1'][0]}:{r['b1'][1]}, p={r['p1']:.3f})"
              f"  Hit@3 {r['h3']:5.1f} ({r['d3']:+5.1f}, {r['b3'][0]}:{r['b3'][1]}, p={r['p3']:.3f})")
