"""Holm families for the no-filter arm, recomputed with the independent functions."""
import runpy, sys, io, contextlib
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    g = runpy.run_path(sys.argv[1] if len(sys.argv) > 1 else
                       str(__import__('pathlib').Path(__file__).resolve().parent / 'verify_independent.py'))            # reuses R, qas, BASES, binomtest, multipletests
R, qas, BASES, binomtest, multipletests, math = g["R"], g["qas"], g["BASES"], g["binomtest"], g["multipletests"], g["math"]
def family(ref, bases):
    T = []
    for doc in ("nasa", "rfc9110", "wells"):
        n = len(qas[doc])
        for base in bases:
            for k in (1, 3):
                a = [(x or 99) <= k for x in R[doc][ref]]; b = [(x or 99) <= k for x in R[doc][base]]
                x = sum(p and not q for p, q in zip(a, b)); y = sum(q and not p for p, q in zip(a, b))
                d = (x - y) / n; se = math.sqrt((x + y) / n - d * d) / math.sqrt(n)
                T.append(dict(doc=doc, base=base, k=k, x=x, y=y, p=binomtest(x, x + y, 0.5).pvalue if x + y else 1.0,
                              d=d * 100, lo=(d - 1.96 * se) * 100, hi=(d + 1.96 * se) * 100))
    rej, ph, _, _ = multipletests([t["p"] for t in T], alpha=0.05, method="holm")
    for t, r, p in zip(T, rej, ph): t["holm"], t["rej"] = p, bool(r)
    return T
for title, ref, bases in [("A) no-filter arm vs all 5 baselines, Holm m=30", "llm_incremental_nofilter", BASES),
                          ("B) no-filter arm vs 4 baselines (as nofilter_v4.py), Holm m=24", "llm_incremental_nofilter", BASES[:-1]),
                          ("C) filtered vs no-filter arm (effect of the filter itself), m=6", "llm_incremental", ["llm_incremental_nofilter"])]:
    print("\n" + title)
    for t in family(ref, bases):
        if t["doc"] == "nasa" or t["rej"] or ref == "llm_incremental":
            print(f"  {t['doc']:8} {t['base']:26} Hit@{t['k']} {t['d']:+5.1f} pp [{t['lo']:+5.1f}, {t['hi']:+5.1f}] {t['x']:3}:{t['y']:<3} p={t['p']:.4f} Holm={t['holm']:.4f}{' *' if t['rej'] else ''}")
