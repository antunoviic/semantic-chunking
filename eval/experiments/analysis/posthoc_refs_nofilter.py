"""POST-HOC sensitivity for the no-filter (tested) arm: nasa without the 35 questions from
sections 40-43 (References Cited / Bibliography). Same code path as [7] in verify_independent.py,
only with llm_incremental_nofilter as reference arm."""
import runpy, sys, io, contextlib, json
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    g = runpy.run_path(sys.argv[1] if len(sys.argv) > 1 else
                       str(__import__('pathlib').Path(__file__).resolve().parent / 'verify_independent.py'))
R, qas, BASES, binomtest, multipletests, math = g["R"], g["qas"], g["BASES"], g["binomtest"], g["multipletests"], g["math"]
REPLIES, norm = g["REPLIES"], g["norm"]
REF = "llm_incremental_nofilter"
def fam(keep_nasa):
    T = []
    for doc in ("nasa", "rfc9110", "wells"):
        idx = keep_nasa if doc == "nasa" else range(len(qas[doc])); idx = list(idx); n = len(idx)
        for base in BASES:
            for k in (1, 3):
                a = [(R[doc][REF][i] or 99) <= k for i in idx]; b = [(R[doc][base][i] or 99) <= k for i in idx]
                x = sum(p and not q for p, q in zip(a, b)); y = sum(q and not p for p, q in zip(a, b))
                d = (x - y) / n; se = math.sqrt((x + y) / n - d * d) / math.sqrt(n)
                T.append(dict(doc=doc, base=base, k=k, x=x, y=y, n=n, p=binomtest(x, x + y, 0.5).pvalue if x + y else 1.0,
                              d=d * 100, lo=(d - 1.96 * se) * 100, hi=(d + 1.96 * se) * 100))
    rej, ph, _, _ = multipletests([t["p"] for t in T], alpha=0.05, method="holm")
    for t, r, p in zip(T, rej, ph): t["holm"], t["rej"] = p, bool(r)
    return T
sec = {norm(r["question"]): r["section"] for r in json.load(open(REPLIES["nasa"]))}
keep = [i for i, p in enumerate(qas["nasa"]) if sec[norm(p["question"])] < 40]
for title, k_ in (("all questions", range(len(qas["nasa"]))), (f"nasa without references (n={len(keep)})", keep)):
    T = fam(k_)
    print(f"\n{title}: {sum(t['rej'] for t in T)} of 30 significant")
    for t in T:
        if t["doc"] == "nasa" or t["rej"]:
            print(f"  {t['doc']:8} {t['base']:24} Hit@{t['k']} {t['d']:+5.1f} pp [{t['lo']:+5.1f}, {t['hi']:+5.1f}] "
                  f"{t['x']:3}:{t['y']:<3} p={t['p']:.4f} Holm={t['holm']:.4f}{' *' if t['rej'] else ''}")
