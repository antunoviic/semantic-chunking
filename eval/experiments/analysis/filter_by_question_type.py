"""How much of the filter's cost comes from questions about the reference lists?
Filtered vs unfiltered LLM arm, all questions and without nasa prompt sections 40-43
(References Cited / Bibliography, excluded by prompt rule 7). Post hoc, descriptive."""
import contextlib, io, json, runpy
from pathlib import Path
HERE = Path(__file__).resolve().parent
with contextlib.redirect_stdout(io.StringIO()):
    g = runpy.run_path(str(HERE / "verify_independent.py"))
R, qas, intact, norm, binomtest, multipletests = g["R"], g["qas"], g["intact"], g["norm"], g["binomtest"], g["multipletests"]
sec = {norm(r["question"]): r["section"] for r in json.load(open("question_prompts/nasa_literal_v4/replies.json"))}
hit = lambda r, k: (r or 99) <= k
rows = []
for label, doc, keep in [("nasa, all 534", "nasa", None), ("nasa, without sections 40-43", "nasa", lambda q: sec[norm(q["question"])] < 40),
                         ("nasa, only sections 40-43", "nasa", lambda q: sec[norm(q["question"])] >= 40),
                         ("rfc9110, all 549", "rfc9110", None), ("wells, all 400", "wells", None)]:
    idx = [i for i, q in enumerate(qas[doc]) if keep is None or keep(q)]
    for k in (1, 3):
        a = [hit(R[doc]["llm_incremental"][i], k) for i in idx]; b = [hit(R[doc]["llm_incremental_nofilter"][i], k) for i in idx]
        x = sum(p and not q for p, q in zip(a, b)); y = sum(q and not p for p, q in zip(a, b))
        lost = sum(intact[doc]["llm_incremental_nofilter"][i] and not intact[doc]["llm_incremental"][i] for i in idx)
        rows.append(dict(label=label, k=k, n=len(idx), x=x, y=y, d=(x - y) * 100 / len(idx), lost=lost,
                         p=binomtest(x, x + y, 0.5).pvalue if x + y else 1.0))
fam = [r for r in rows if r["label"] in ("nasa, without sections 40-43", "rfc9110, all 549", "wells, all 400")]
rej, ph, _, _ = multipletests([r["p"] for r in fam], alpha=0.05, method="holm")
for r, p_ in zip(fam, ph): r["holm_alt"] = p_
fam0 = [r for r in rows if r["label"] in ("nasa, all 534", "rfc9110, all 549", "wells, all 400")]
rej0, ph0, _, _ = multipletests([r["p"] for r in fam0], alpha=0.05, method="holm")
for r, p_ in zip(fam0, ph0): r["holm_main"] = p_
for r in rows:
    print(f"{r['label']:30} n={r['n']:3} Hit@{r['k']}  filter {r['d']:+5.1f} pp  gained:lost {r['x']:2}:{r['y']:<2} p={r['p']:.4f}"
          f"  Holm(main)={r.get('holm_main', float('nan')):.4f}  Holm(w/o refs)={r.get('holm_alt', float('nan')):.4f}  anchors lost {r['lost']}")
