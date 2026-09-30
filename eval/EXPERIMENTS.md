# Experiments

The measurements behind the bachelor thesis: the scripts that produced the
results, and the reports they wrote.

Three directories in this repository sound similar, so to be explicit about
which is which:

| | What it is for |
|---|---|
| `README.md` (project root) | The library: how to chunk a document with it. |
| `tools/` (project root) | A small toolkit for *your own* documents — building a question set and checking a document is worth evaluating. |
| `eval/experiments/` (here) | The experiments of this thesis. Not meant to be reused on other documents. |

Everything here answers *"does LLM-based chunking retrieve better than
rule-based chunking, and under what conditions"* — not *"how do I chunk my
document"*. For the latter, see the project README.

```
eval/experiments/          the experiments and the analyses they feed
eval/experiments/analysis/ the final tables of the thesis, their independent
                           recomputation, and the demo of the defence
thesis/thesis.pdf          the thesis itself
thesis/presentation.pdf    the slides of the defence
thesis/latex/              the LaTeX source of the thesis
thesis/results/            where the scripts write their reports — not tracked
```

The reports the scripts write are not in this repository, but the scripts are,
so every report can be regenerated from them. The numbers that reached the
thesis are in chapter 4 of `thesis/thesis.pdf`, and
`analysis/verify_independent.py` recomputes them from the tracked chunk caches
and question sets.

## Running them

All scripts are invoked **from the project root**, never from inside
`eval/experiments/`.
They read chunk caches from `chunks_cache/`, question sets from `eval_cache/`
and documents from `docs/`, all relative to the root. The LLM chunks of the
reported run are tracked in `chunks_cache/`, so the final comparison reruns
after a clone with only the embedding model; the prompts behind the question
sets are in `question_prompts/`.

```bash
bash   eval/experiments/run_v4.sh                      # the full run
python eval/experiments/check_heading_spans.py docs/rfc9110.txt
python eval/experiments/run_significance.py --document docs/wells.txt \
       --questions eval_cache/wells_questions.json --label wells
```

## The scripts

| Script | Produces / does | Needs the model? |
|---|---|---|
| `run_v4.sh` | The chunking run behind the thesis: four arms on each of the three documents, plus `--midpoint-split` on nasa alone, then the first evaluation (approximate HNSW search). Resumable — a finished arm is skipped, a stale one re-chunked. Closes with a provenance check over exactly the arms it expects. | yes, hours |
| `run_fair_baselines.py` | The final comparison: the cached LLM arms against baselines held to the same terms — recursive with sentence ends, the same chunker with the model replaced by an always-YES client, and a SemanticChunker tuned to the same length and cap — with exact cosine search, McNemar and Holm. `--match filtered` matches to the filtered LLM arm (714 / 720 / 725 characters), the setting reported in the thesis. | embedding only |
| `analyze_fair_arms.py` | Robustness of that comparison: token-level metrics, other hit thresholds, and how often chunk boundaries fall on section headings. | no |
| `enrich_arm.py` | Derives the `[Topic: ...]` arm from an already chunked one. Called by `run_v4.sh`; enrichment is post-processing, so the boundaries are reused rather than recomputed. | yes |
| `run_determinism.py` | Chunks one document N times and compares. The basis for using a fixed seed instead of averaging over runs. | yes |
| `run_child_ablation.py` | Sensitivity of the parent-child result to the child size (150 / 250 / 400 characters). Reads the cached `incremental` arm and derives the children arithmetically, so no boundary is recomputed. | embedding only |
| `run_index_robustness.py` | Default HNSW search against a near-exhaustive one, to bound how much of a measured difference is an artefact of the approximate index. One report per document. | embedding only |
| `run_significance.py` | Paired McNemar tests against the reference arm, on Hit@1 rather than Hit@10 only. | embedding only |
| `check_heading_spans.py` | How many chunks run across a heading, per strategy — the direct answer to the supervisor's observation. | no |
| `test_heading_detection.py` | Agreement between the LLM heading detector and the regex baseline, measured before the two are compared as chunking arms. | yes |
| `prepare_gutenberg.py` | Turns a raw Project Gutenberg text into an evaluation document: strips boilerplate, captions and index, keeps chapter headings. Produced `docs/wells.txt`. | no |

### The final tables (`analysis/`)

The numbers of the thesis chapter on the evaluation come from these scripts, run
in this order from the project root. Their outputs and embedding caches go to
`eval_results/analysis/`, which is not tracked.

| Script | Produces / does | Needs the model? |
|---|---|---|
| `run_fair_baselines.py --match filtered --docs wells` (above) | Step 1: the length-matched baselines for wells (725 characters), whose single question set is reused by step 2. | embedding only |
| `analysis/run_v4_eval.py` | Step 2: the v4 question sets of nasa and rfc9110 against the same arms; writes `eval_results/fair_baselines_v4.json`. | embedding only |
| `analysis/final_table_v4_nofilter.py` | Step 3, the main result: the LLM arm without the low-information filter against the five baselines, 30 McNemar tests with Holm correction. | no |
| `analysis/final_table_v4.py` | The same for the arm with the filter. | no |
| `analysis/verify_independent.py` | Recomputes every hit rate and test with its own search, hit criterion and statistics libraries; `nofilter_family.py`, `posthoc_refs_nofilter.py` and `filter_by_question_type.py` build on it. | spot check only |
| `analysis/parent_child_v4_nofilter.py` | Parent-child retrieval applied to every strategy. | embedding only |
| `analysis/ablations_v4.py` | The ablation arms (split point, heading detection, topic labels) on the v4 question sets. | embedding only |
| `analysis/make_forest_v4_nofilter.py` | The figure of the paired differences with their intervals. | no |
| `analysis/demo_evaluation.py` | The live demo: evaluates the cached chunks and prints the result table in a few seconds. | no |

The pipeline reads chunk caches through `ChunkCache.load()`, which
compares the cache's code fingerprint against the running code and reports a
mismatched cache as absent. An arm from an older code state therefore cannot
enter a comparison unnoticed — it is skipped, loudly. The exception is
`run_fair_baselines.py`, which opens the two LLM caches directly and records their
fingerprints in its output instead; `analyze_fair_arms.py` in turn reads what it
saved, so the arms it analyses are never re-chunked.

**Run these under Python 3.9.** The reported run is stamped `3c3df72cca46`,
which was computed under 3.9. The fingerprint hashes `ast.dump()` of the modules
that decide boundaries, and that text form is not stable across Python minor
versions: the same unchanged source yields `c432af2e9c46` under 3.13. Under a
different interpreter every cache of the reported run is therefore rejected as
foreign, and `run_v4.sh` would re-chunk all of it — roughly three days of model
time. Caches now carry the interpreter alongside the digest, so `load()` names
that case instead of letting a switched Python look like an edited chunker.
