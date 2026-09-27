"""Step 2 of the final evaluation: the v4 question sets for nasa and rfc9110 against the
cached LLM arms and the length-matched baselines (thesis/scripts/run_fair_baselines.py).
wells keeps its single question set, taken from eval_results/fair_baselines_filtered.json,
which `python thesis/scripts/run_fair_baselines.py --match filtered` writes (step 1).
Writes eval_results/fair_baselines_v4.json. Run from the repository root."""
import json, sys
sys.path.insert(0, "."); sys.path.insert(0, "thesis/scripts")
from pathlib import Path
from run_fair_baselines import CachedEmbedder, run_document, write_report, REFERENCES

embed = CachedEmbedder(Path("eval_results/fair_baselines_embeddings.pkl"))
results = {}
for label, doc, qfile in (("nasa", "docs/nasa.pdf", "eval_cache/nasa_questions_literal_v4.json"),
                          ("rfc9110", "docs/rfc9110.txt", "eval_cache/rfc9110_questions_v4.json")):
    results[label] = run_document(label, doc, qfile, embed, "filtered")
    embed.save()

old = json.load(open("eval_results/fair_baselines_filtered.json"))["results"]
results["wells"] = old["wells"]  # unchanged, per plan

tests = write_report(results, Path("eval_results/analysis/FAIR_BASELINES_v4.md"), REFERENCES["filtered"])
Path("eval_results/fair_baselines_v4.json").write_text(json.dumps(
    {"match": "filtered", "reference": REFERENCES["filtered"], "results": results, "tests": tests},
    ensure_ascii=False))
print("\nDONE")
