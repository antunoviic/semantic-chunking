# Experiments

The measurements behind the bachelor thesis: the scripts that produced the
results, and the reports they wrote.

Three directories in this repository sound similar, so to be explicit about
which is which:

| | What it is for |
|---|---|
| `README.md` (project root) | The library: how to chunk a document with it. |
| `tools/` (project root) | A small toolkit for *your own* documents — building a question set and checking a document is worth evaluating. |
| `thesis/` (here) | The experiments of this thesis. Not meant to be reused on other documents. |

Everything here answers *"does LLM-based chunking retrieve better than
rule-based chunking, and under what conditions"* — not *"how do I chunk my
document"*. For the latter, see the project README.

```
thesis/
  scripts/   the experiments and the analyses they feed
  results/   where they write their reports — not tracked, see below
```

Neither the LaTeX source of the thesis nor the reports are in this repository.
The scripts are, so every report can be regenerated from them.

## Running them

All scripts are invoked **from the project root**, never from inside `thesis/`.
They read chunk caches from `chunks_cache/`, question sets from `eval_cache/`
and documents from `docs/`, all relative to the root:

```bash
bash   thesis/scripts/run_v4.sh                      # the full run
python thesis/scripts/check_heading_spans.py docs/rfc9110.txt
python thesis/scripts/run_significance.py --document docs/wells.txt \
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

## The reports

`results/` is where the scripts write, and it is deliberately not tracked. It
holds working notes in German towards the thesis chapters — an analysis plan, a
record of every design decision, and measurement reports from several runs, only
some of which are current. Published together with the package they would invite
the wrong reading: a table whose header names a superseded question set looks
like a result, not like a note to self.

The reports are regenerated by running the scripts above. Two of them are
documents rather than measurements and exist for reasons worth stating here:

- **`TEST_PROTOCOL.md`** — the analysis plan, fixed on 2026-09-14 *before* the
  final run: which arms, which metrics, which tests, and which questions were
  dropped. It exists so that the choice of what to report cannot follow from
  what the numbers turned out to be.
- **`DESIGN_DECISIONS.md`** — every design choice with its reasoning, its
  measured effect and its known weaknesses, including the assumptions the
  measurements refuted and what is still open.

Every measurement report carries its date and its question set in the header, so
a `_v2` result is never mistaken for a `_v3` one. Both are cited in the thesis
itself, which is where they belong.
