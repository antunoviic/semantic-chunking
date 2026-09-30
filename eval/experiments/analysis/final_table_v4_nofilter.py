"""v4 question sets for nasa/rfc9110, wells unchanged. Step 3: 30 paired tests (five baselines, three
documents, Hit@1 and Hit@3), McNemar exact, Holm."""
import json, math, sys
from pathlib import Path
sys.path.insert(0, "."); sys.path.insert(0, "eval/experiments")
from run_fair_baselines import CachedEmbedder, normalized, ranks, anchors_intact, mid_sentence, hits, holm
from run_significance import mcnemar_p
from app.document_reader import DocumentReader
from eval.strategy_evaluator import build_strategies

embed = CachedEmbedder(Path("eval_results/fair_baselines_embeddings.pkl"))
v4 = json.load(open("eval_results/fair_baselines_v4.json"))["results"]

DOCS = {"nasa": "docs/nasa.pdf", "rfc9110": "docs/rfc9110.txt", "wells": "docs/wells.txt"}
Q = {"nasa": "eval_cache/nasa_questions_literal_v4.json", "rfc9110": "eval_cache/rfc9110_questions_v4.json",
     "wells": "eval_cache/wells_questions.json"}
ARMS = ["llm_incremental", "llm_incremental_nofilter", "packing_matched", "recursive_sent_matched",
        "semantic_matched", "recursive_matched", "fixed_matched"]

out, tests = {}, []
for doc, path in DOCS.items():
    r = v4[doc]; qa = json.load(open(Q[doc])); n = len(qa); t = r["target"]
    R = {a: r["ranks"][a] for a in ARMS if a in r["ranks"]}
    fixed = build_strategies(DocumentReader(path).extract_text(), match_len=t)[f"fixed_matched_{t}"]
    q = normalized(embed([p["question"] for p in qa]))
    R["fixed_matched"], _ = ranks(fixed, q, qa, embed); embed.save()
    chunks = dict(r["chunks"]); chunks["fixed_matched"] = fixed
    out[doc] = {a: dict(h1=sum(hits(R[a], 1)) * 100 / n, h3=sum(hits(R[a], 3)) * 100 / n,
                        mean=round(sum(map(len, chunks[a])) / len(chunks[a])),
                        mid=mid_sentence(chunks[a]), intact=anchors_intact(chunks[a], qa)) for a in ARMS}
    for ref in ("llm_incremental_nofilter",):
        for base in ARMS[2:]:
            for k in (1, 3):
                a, b = hits(R[ref], k), hits(R[base], k)
                x = sum(p and not c for p, c in zip(a, b)); y = sum(c and not p for p, c in zip(a, b))
                d = (x - y) / n; se = math.sqrt(max(x + y - (x - y) ** 2 / n, 0)) / n
                tests.append(dict(ref=ref, doc=doc, base=base, k=k, n_llm=x, n_base=y, d=d*100,
                                  lo=(d-1.96*se)*100, hi=(d+1.96*se)*100, p=mcnemar_p(x, y)))
for doc, rows in out.items():
    print(f"\n{doc}  (n={len(json.load(open(Q[doc])))})")
    for a, v in rows.items():
        print(f"  {a:26} mean {v['mean']:4}  Hit@1 {v['h1']:5.1f}  Hit@3 {v['h3']:5.1f}  mid-sentence {v['mid']:5.1f}%  anchors intact {v['intact']:5.1f}%")
holm(tests)
print(f"\nTests: llm_incremental vs each baseline, Holm over m = {len(tests)}")
for t in tests:
    print(f"  {t['doc']:8} {t['base']:24} Hit@{t['k']}  {t['d']:+5.1f} pp [{t['lo']:+5.1f}, {t['hi']:+5.1f}]  {t['n_llm']:3}:{t['n_base']:<3} p={t['p']:.4f}  Holm={t['p_holm']:.4f} {'*' if t['reject'] else ''}")
json.dump({"arms": out, "tests": tests}, open("eval_results/final_table_v4_nofilter.json", "w"), indent=1)
