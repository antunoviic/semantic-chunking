"""Independent re-computation of the v4 evaluation (final_table_v4.json).

Shares with the original pipeline only the *inputs*: document extraction
(DocumentReader), the rule-based fixed splitter (build_strategies), the chunk
lists and the embedding cache. Everything that turns inputs into numbers is
re-implemented here: cosine search (float64), the 80 % hit criterion (window
containment instead of difflib), anchors intact, McNemar (scipy binomtest),
Holm (statsmodels), Wald CIs.
Run from the repo root:  python <this file>
"""
import hashlib, json, math, pickle, random, re, sys, urllib.request
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT))
from app.document_reader import DocumentReader          # noqa: E402  (input only)
from eval.strategy_evaluator import build_strategies     # noqa: E402  (input only)
from scipy.stats import binomtest                         # noqa: E402
from statsmodels.stats.multitest import multipletests     # noqa: E402

DOCS = {"nasa": "docs/nasa.pdf", "rfc9110": "docs/rfc9110.txt", "wells": "docs/wells.txt"}
QF = {"nasa": "eval_cache/nasa_questions_literal_v4.json",
      "rfc9110": "eval_cache/rfc9110_questions_v4.json",
      "wells": "eval_cache/wells_questions.json"}
REPLIES = {"nasa": "question_prompts/nasa_literal_v4/replies.json",
           "rfc9110": "question_prompts/rfc9110_literal_v4/replies.json"}
ARMS = ["llm_incremental", "llm_incremental_nofilter", "packing_matched", "recursive_sent_matched",
        "semantic_matched", "recursive_matched", "fixed_matched"]
BASES = ARMS[2:]
TOPIC = re.compile(r"^\[Topic:[^\]]*\]\s*")
norm = lambda s: " ".join(s.split())

final = json.load(open("eval_results/final_table_v4.json"))
v4 = json.load(open("eval_results/fair_baselines_v4.json"))["results"]
v3 = json.load(open("eval_results/fair_baselines_filtered.json"))["results"]
cache = pickle.loads(Path("eval_results/fair_baselines_embeddings.pkl").read_bytes())
problems = []


def say(msg, ok=True):
    print(("  ok   " if ok else "  FAIL ") + msg)
    if not ok:
        problems.append(msg)


def vecs(texts):
    keys = [hashlib.sha1(t.encode("utf-8")).hexdigest() for t in texts]
    missing = [t for t, k in zip(texts, keys) if k not in cache]
    if missing:
        raise SystemExit(f"{len(missing)} texts not in the embedding cache")
    m = np.vstack([cache[k] for k in keys]).astype(np.float64)
    return m / np.linalg.norm(m, axis=1, keepdims=True)


def covered(anchor, chunk):
    """True iff some contiguous run of >= 80 % of the anchor occurs in the chunk."""
    a, b = norm(anchor), norm(TOPIC.sub("", chunk.strip()))
    if not a or not b:
        return False
    m = -(-len(a) * 4 // 5)                     # ceil(0.8 * len)
    return any(a[i:i + m] in b for i in range(len(a) - m + 1))


def first_hit_rank(chunks, Q, qa, kmax=3):
    S = Q @ vecs(chunks).T
    order = np.argsort(-S, axis=1, kind="stable")[:, :kmax]
    ranks, near_ties = [], 0
    for qi, (row, pair) in enumerate(zip(order, qa)):
        s = np.sort(S[qi])[::-1][:kmax + 1]
        near_ties += bool(np.any(np.abs(np.diff(s)) < 1e-6))
        ranks.append(next((j for j, i in enumerate(row, 1) if covered(pair["source_text"], chunks[i])), None))
    return ranks, near_ties


# --------------------------------------------------------------------------- 1
print("\n[1] question sets: every anchor verbatim, unique in the document, >= 40 chars, no duplicates")
texts, qas = {}, {}
for doc, path in DOCS.items():
    texts[doc] = DocumentReader(path).extract_text()
    tn = norm(texts[doc])
    qa = json.load(open(QF[doc])); qas[doc] = qa
    anchors = [norm(p["source_text"]) for p in qa]
    bad_verbatim = sum(a not in tn for a in anchors)
    bad_unique = sum(tn.count(a) != 1 for a in anchors)
    short = sum(len(p["source_text"]) < 40 for p in qa)
    dup_a = len(anchors) - len(set(anchors))
    dup_q = len(qa) - len({norm(p["question"]) for p in qa})
    lens = [len(p["source_text"]) for p in qa]
    say(f"{doc:8} n={len(qa)}  verbatim-miss={bad_verbatim} not-unique={bad_unique} short={short} "
        f"dup-anchor={dup_a} dup-question={dup_q}  anchor length {min(lens)}..{max(lens)}",
        not (bad_verbatim or bad_unique or short or dup_a or dup_q))
    if doc in REPLIES:
        rep = json.load(open(REPLIES[doc]))
        rq = {norm(r["question"]) for r in rep}
        say(f"{doc:8} {len(rep)} replies -> {len(qa)} kept; every kept question comes from the replies",
            all(norm(p["question"]) in rq for p in qa))
tot = sum(len(q) for q in qas.values())
allens = [len(p["source_text"]) for q in qas.values() for p in q]
print(f"       total {tot} questions, anchors {min(allens)}..{max(allens)} chars")

# --------------------------------------------------------------------------- 2
print("\n[2] chunk provenance")
chunks = {}
for doc in DOCS:
    stem = Path(DOCS[doc]).stem
    chunks[doc] = dict(v4[doc]["chunks"])
    for arm, suffix in (("llm_incremental", "incremental"), ("llm_incremental_nofilter", "incremental_nofilter")):
        c = json.load(open(f"chunks_cache/{stem}_{suffix}.json"))
        say(f"{doc:8} {arm:26} == chunks_cache/{stem}_{suffix}.json  (code_digest {c.get('code_digest', '?')[:12]})",
            c["chunks"] == chunks[doc][arm])
    if doc != "wells" and doc in v3:            # only when step 1 also ran this document
        for arm in BASES[:-1]:
            say(f"{doc:8} {arm:26} v4 run == v3 run (baselines do not depend on the questions)",
                v3[doc]["chunks"][arm] == chunks[doc][arm])
    t = v4[doc]["target"]
    llm_mean = round(sum(map(len, chunks[doc]["llm_incremental"])) / len(chunks[doc]["llm_incremental"]))
    say(f"{doc:8} target {t} == mean length of filtered LLM arm ({llm_mean})", t == llm_mean)
    chunks[doc]["fixed_matched"] = build_strategies(texts[doc], match_len=t)[f"fixed_matched_{t}"]
    for arm in BASES:
        mean = sum(map(len, chunks[doc][arm])) / len(chunks[doc][arm])
        dev = (mean - t) / t * 100
        print(f"       {doc:8} {arm:26} {len(chunks[doc][arm]):5} chunks  mean {mean:7.1f}  ({dev:+.1f} % vs target)"
              f"  max {max(map(len, chunks[doc][arm]))}")

# --------------------------------------------------------------------------- 3
print("\n[3] embedding cache spot check (re-embed 24 random cached texts with bge-m3 via Ollama)")
random.seed(0)
sample = random.sample([c for d in DOCS for a in ARMS for c in chunks[d][a]], 20) + \
         random.sample([p["question"] for q in qas.values() for p in q], 4)
try:
    req = urllib.request.Request("http://localhost:11434/api/embed", method="POST",
                                 data=json.dumps({"model": "bge-m3", "input": [s[:5000] for s in sample]}).encode(),
                                 headers={"Content-Type": "application/json"})
    fresh = np.array(json.load(urllib.request.urlopen(req, timeout=300))["embeddings"], dtype=np.float64)
    fresh /= np.linalg.norm(fresh, axis=1, keepdims=True)
    cos = (fresh * vecs(sample)).sum(1)
    say(f"cosine(cached, fresh): min {cos.min():.6f}  mean {cos.mean():.6f}", cos.min() > 0.999)
except Exception as e:                                   # Ollama not running
    say(f"spot check skipped: {e}", False)

# --------------------------------------------------------------------------- 4
print("\n[4] Hit@1 / Hit@3 / anchors intact / mean length vs final_table_v4.json")
R, intact = {}, {}
for doc in DOCS:
    qa = qas[doc]; n = len(qa)
    Q = vecs([p["question"] for p in qa])
    R[doc], intact[doc] = {}, {}
    for arm in ARMS:
        r, ties = first_hit_rank(chunks[doc][arm], Q, qa)
        R[doc][arm] = r
        blob = chunks[doc][arm]
        nb = [norm(TOPIC.sub("", c.strip())) for c in blob]
        flags = []
        for p in qa:
            a = norm(p["source_text"]); m = -(-len(a) * 4 // 5)
            core = a[len(a) - m:m]              # every window of length m contains this
            flags.append(any(covered(a, c) for c in nb if core in c))
        intact[doc][arm] = flags
        h1 = sum(x is not None and x <= 1 for x in r) * 100 / n
        h3 = sum(x is not None and x <= 3 for x in r) * 100 / n
        it = sum(intact[doc][arm]) * 100 / n
        mean = round(sum(map(len, blob)) / len(blob))
        f = final["arms"][doc][arm]
        same = abs(h1 - f["h1"]) < 1e-9 and abs(h3 - f["h3"]) < 1e-9 and abs(it - f["intact"]) < 0.05 and mean == f["mean"]
        say(f"{doc:8} {arm:26} Hit@1 {h1:5.1f} ({f['h1']:5.1f})  Hit@3 {h3:5.1f} ({f['h3']:5.1f})  "
            f"intact {it:5.1f} ({f['intact']:5.1f})  mean {mean} ({f['mean']})  near-ties {ties}", same)

# --------------------------------------------------------------------------- 5
print("\n[5] 30 paired tests, llm_incremental vs each baseline: McNemar exact, Holm, Wald 95 % CI")
tests = []
for doc in DOCS:
    n = len(qas[doc])
    for base in BASES:
        for k in (1, 3):
            a = [x is not None and x <= k for x in R[doc]["llm_incremental"]]
            b = [x is not None and x <= k for x in R[doc][base]]
            x = sum(p and not q for p, q in zip(a, b)); y = sum(q and not p for p, q in zip(a, b))
            p = binomtest(x, x + y, 0.5).pvalue if x + y else 1.0
            d = (x - y) / n; se = math.sqrt((x + y) / n - d * d) / math.sqrt(n)
            tests.append(dict(doc=doc, base=base, k=k, x=x, y=y, p=p, d=d * 100, lo=(d - 1.96 * se) * 100,
                              hi=(d + 1.96 * se) * 100))
rej, ph, _, _ = multipletests([t["p"] for t in tests], alpha=0.05, method="holm")
ref = {(t["doc"], t["base"], t["k"]): t for t in final["tests"]}
for t, r_, p_ in zip(tests, rej, ph):
    f = ref[(t["doc"], t["base"], t["k"])]
    same = (t["x"], t["y"]) == (f["n_llm"], f["n_base"]) and abs(t["p"] - f["p"]) < 1e-9 \
        and abs(p_ - f["p_holm"]) < 1e-9 and bool(r_) == f["reject"] and abs(t["lo"] - f["lo"]) < 1e-9
    say(f"{t['doc']:8} {t['base']:24} Hit@{t['k']} {t['d']:+5.1f} pp [{t['lo']:+5.1f}, {t['hi']:+5.1f}] "
        f"{t['x']:3}:{t['y']:<3} p={t['p']:.4f} Holm={p_:.4f}{' *' if r_ else '  '}", same)

# --------------------------------------------------------------------------- 6
print("\n[6] anchors the low-information filter removed (intact without filter, not with filter)")
for doc in ("nasa", "rfc9110"):
    qa = qas[doc]
    lost = [i for i in range(len(qa)) if intact[doc]["llm_incremental_nofilter"][i] and not intact[doc]["llm_incremental"][i]]
    base_ok = {b: sum(intact[doc][b][i] for i in lost) for b in BASES}
    print(f"  {doc}: {len(lost)} of {len(qa)} anchors lost by the filter; intact in baselines: {base_ok}")
    if doc in REPLIES:
        sec = {norm(r["question"]): r.get("section") for r in json.load(open(REPLIES[doc]))}
        print(f"    by prompt section: {dict(sorted(Counter(sec.get(norm(qa[i]['question'])) for i in lost).items()))}")
    for i in lost[:60]:
        print(f"    - {qa[i]['question'][:95]}")
    # how many of the filtered arm's Hit@3 misses are explained by a lost anchor
    miss3 = [i for i in range(len(qa)) if not (R[doc]['llm_incremental'][i] or 99) <= 3]
    print(f"    filtered arm misses Hit@3 on {len(miss3)} questions, {len(set(miss3) & set(lost))} of them have a lost anchor")

print("\nSUMMARY:", "all checks passed" if not problems else f"{len(problems)} problem(s)")
for p in problems:
    print("  -", p)

# --------------------------------------------------------------------------- 7
print("\n[7] POST-HOC sensitivity, not the pre-committed analysis: nasa without the 35 questions from")
print("    sections 40-43 (References Cited / Bibliography), which prompt rule 7 excluded")
sec = {norm(r["question"]): r["section"] for r in json.load(open(REPLIES["nasa"]))}
keep = [i for i, p in enumerate(qas["nasa"]) if sec[norm(p["question"])] < 40]
n = len(keep)
alt = [t for t in tests if t["doc"] != "nasa"]
for base in BASES:
    for k in (1, 3):
        a = [(R["nasa"]["llm_incremental"][i] or 99) <= k for i in keep]
        b = [(R["nasa"][base][i] or 99) <= k for i in keep]
        x = sum(p and not q for p, q in zip(a, b)); y = sum(q and not p for p, q in zip(a, b))
        d = (x - y) / n; se = math.sqrt((x + y) / n - d * d) / math.sqrt(n)
        alt.append(dict(doc="nasa*", base=base, k=k, x=x, y=y, p=binomtest(x, x + y, 0.5).pvalue,
                        d=d * 100, lo=(d - 1.96 * se) * 100, hi=(d + 1.96 * se) * 100))
rej, ph, _, _ = multipletests([t["p"] for t in alt], alpha=0.05, method="holm")
for doc in ("nasa*",):
    for arm in ARMS:
        h1 = sum((R["nasa"][arm][i] or 99) <= 1 for i in keep) * 100 / n
        h3 = sum((R["nasa"][arm][i] or 99) <= 3 for i in keep) * 100 / n
        print(f"  n={n} {arm:26} Hit@1 {h1:5.1f}  Hit@3 {h3:5.1f}")
for t, r_, p_ in zip(alt, rej, ph):
    if t["doc"] == "nasa*" or r_:
        print(f"  {t['doc']:8} {t['base']:24} Hit@{t['k']} {t['d']:+5.1f} pp [{t['lo']:+5.1f}, {t['hi']:+5.1f}] "
              f"{t['x']:3}:{t['y']:<3} p={t['p']:.4f} Holm={p_:.4f}{' *' if r_ else ''}")
