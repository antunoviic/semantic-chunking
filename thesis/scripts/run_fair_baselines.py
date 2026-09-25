"""Does the model's topic decision add anything a rule cannot?

Every arm is held to the same terms, so the only difference left is who draws
the boundaries:

  llm_incremental_nofilter   the v4 LLM arm (cached), no low-information filter
  packing_matched            the same chunker with the model replaced by a
                             client that always answers YES — only the cap cuts
  recursive_matched          the thesis baseline (LangChain's default separators)
  recursive_sent_matched     the same splitter, sentence ends before spaces
  semantic_matched           LangChain's SemanticChunker, threshold tuned, same cap

By default the baselines are length-matched to the mean of the unfiltered LLM
arm, which is tested; the filtered arm is only reported, because it searches a
smaller corpus. That mean is pulled down by many tiny leftover chunks, while a
typical LLM chunk is nearer the filtered arm's mean — so `--match filtered`
matches to that mean instead and tests the filtered arm.

Retrieval is an exact cosine search over the same bge-m3 embeddings — no HNSW,
so the numbers are deterministic. Hit = the thesis criterion (80 % coverage).
Tests: exact McNemar, the tested LLM arm against each baseline, Hit@1 and
Hit@3, Holm over the whole family.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import re
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from app.chunk_cache import _code_digest
from app.document_reader import DocumentReader
from eval.strategy_evaluator import (StrategyEvaluator, build_packing,
                                     build_recursive_sentences, build_semantic_lc,
                                     build_strategies)
from eval.vectorstore import OllamaEmbeddingFunction
from run_significance import mcnemar_p

DOCS = {
    "nasa":    ("docs/nasa.pdf",    "eval_cache/nasa_questions_literal_v3.json"),
    "rfc9110": ("docs/rfc9110.txt", "eval_cache/rfc9110_questions_v3.json"),
    "wells":   ("docs/wells.txt",   "eval_cache/wells_questions.json"),
}
REFERENCES = {"unfiltered": "llm_incremental_nofilter", "filtered": "llm_incremental"}
BASELINES = ("packing_matched", "recursive_matched", "recursive_sent_matched", "semantic_matched")
REPORT_KS = (1, 3, 10)
TESTED_KS = (1, 3)
ALPHA = 0.05


class CachedEmbedder:
    """The evaluation's embedding function with a disk cache keyed by text."""

    def __init__(self, path: Path) -> None:
        self._fn = OllamaEmbeddingFunction()
        self._path = path
        self._cache: dict[str, np.ndarray] = (
            pickle.loads(path.read_bytes()) if path.exists() else {})

    @staticmethod
    def _key(text: str) -> str:
        return hashlib.sha1(text.encode("utf-8")).hexdigest()

    def __call__(self, texts: list[str]) -> list[list[float]]:
        keys = [self._key(t) for t in texts]
        missing = {k: t for k, t in zip(keys, texts) if k not in self._cache}
        if missing:
            print(f"      embedding {len(missing)} new texts", flush=True)
            vectors = self._fn(list(missing.values()))
            for k, v in zip(missing, vectors):
                self._cache[k] = np.asarray(v, dtype=np.float32)
        return [self._cache[k] for k in keys]

    def save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_bytes(pickle.dumps(self._cache))


def normalized(vectors) -> np.ndarray:
    m = np.vstack(vectors).astype(np.float32)
    return m / np.linalg.norm(m, axis=1, keepdims=True).clip(min=1e-12)


def ranks(chunks: list[str], q_matrix: np.ndarray, qa: list[dict],
          embed) -> tuple[list[int | None], list[list[int]]]:
    """Rank of the first hit, and the chunk indices of the top 10, per question."""
    order = np.argsort(-(q_matrix @ normalized(embed(chunks)).T), axis=1, kind="stable")
    top = order[:, :max(REPORT_KS)].tolist()
    out: list[int | None] = []
    for row, pair in zip(top, qa):
        out.append(next((j for j, i in enumerate(row, 1)
                         if StrategyEvaluator._is_hit(pair["source_text"], chunks[i])), None))
    return out, top


def anchors_intact(chunks: list[str], qa: list[dict]) -> float:
    """Share of anchors that at least one chunk covers to 80 % — the ceiling.

    Coverage >= 80 % means a common substring of that length exists, i.e. some
    window of the anchor of that length occurs inside one chunk.
    """
    norm = lambda s: " ".join(s.split())
    blob = "\x00".join(map(norm, chunks))
    ok = 0
    for pair in qa:
        a = norm(pair["source_text"])
        m = -(-len(a) * 8 // 10)
        ok += any(a[i:i + m] in blob for i in range(len(a) - m + 1))
    return ok * 100 / len(qa)


def mid_sentence(chunks: list[str]) -> float:
    """Share of chunks that end without sentence-final punctuation."""
    return sum(not re.search(r"[.!?:;)\"']\s*$", c.strip()) for c in chunks) * 100 / len(chunks)


def hits(rr: list[int | None], k: int) -> list[bool]:
    return [r is not None and r <= k for r in rr]


def load_cached_arm(stem: str, suffix: str) -> tuple[list[str], dict]:
    data = json.loads(Path(f"chunks_cache/{stem}_{suffix}.json").read_text(encoding="utf-8"))
    return data["chunks"], data


def run_document(label: str, doc: str, qfile: str, embed: CachedEmbedder, match: str) -> dict:
    stem = Path(doc).stem
    text = DocumentReader(doc).extract_text()
    qa = json.loads(Path(qfile).read_text(encoding="utf-8"))

    reference, ref_data = load_cached_arm(stem, "incremental_nofilter")
    filtered, filt_data = load_cached_arm(stem, "incremental")
    matched = reference if match == "unfiltered" else filtered
    target = round(sum(map(len, matched)) / len(matched))
    cap = ref_data["params"]["max_chunk_chars"]
    print(f"\n=== {label}: {len(qa)} questions, target mean {target} chars, cap {cap}", flush=True)

    arms: dict[str, list[str]] = {"llm_incremental_nofilter": reference, "llm_incremental": filtered}
    print("  building packing_matched", flush=True)
    arms["packing_matched"] = build_packing(text, target)
    arms["recursive_matched"] = build_strategies(text, match_len=target)[f"recursive_matched_{target}"]
    arms["recursive_sent_matched"] = build_recursive_sentences(text, target)
    print("  building semantic_matched (embeds every sentence window)", flush=True)
    arms["semantic_matched"] = build_semantic_lc(text, embed, match_len=target, max_chars=cap)
    embed.save()

    q_matrix = normalized(embed([p["question"] for p in qa]))
    rows, all_ranks, all_top = {}, {}, {}
    for name, chunks in arms.items():
        print(f"  scoring {name} ({len(chunks)} chunks)", flush=True)
        rr, top = ranks(chunks, q_matrix, qa, embed)
        embed.save()
        all_ranks[name], all_top[name] = rr, top
        lens = [len(c) for c in chunks]
        rows[name] = {
            "chunks": len(chunks),
            "mean_len": round(sum(lens) / len(lens)),
            "median_len": int(np.median(lens)),
            "corpus_chars": sum(lens),
            **{f"hit@{k}": round(sum(hits(rr, k)) * 100 / len(qa), 1) for k in REPORT_KS},
            "mrr@10": round(sum(1 / r for r in rr if r) / len(qa), 3),
            "mid_sentence_pct": round(mid_sentence(chunks), 1),
            "anchors_intact_pct": round(anchors_intact(chunks, qa), 1),
        }
    return {"n": len(qa), "target": target, "cap": cap, "rows": rows, "ranks": all_ranks,
            "top10": all_top, "chunks": arms, "document": doc, "questions": qfile,
            "llm_digests": sorted({ref_data.get("code_digest"), filt_data.get("code_digest")})}


def holm(tests: list[dict]) -> None:
    m = len(tests)
    running, stopped = 0.0, False
    for i, t in enumerate(sorted(tests, key=lambda t: t["p"])):
        running = max(running, (m - i) * t["p"])
        t["p_holm"] = min(1.0, running)
        t["reject"] = not stopped and t["p_holm"] <= ALPHA
        stopped = stopped or not t["reject"]


def write_report(results: dict, out: Path, reference: str) -> list[dict]:
    tests = []
    for label, res in results.items():
        ref = res["ranks"][reference]
        for base in BASELINES:
            for k in TESTED_KS:
                a, b = hits(ref, k), hits(res["ranks"][base], k)
                n_a = sum(x and not y for x, y in zip(a, b))
                n_b = sum(y and not x for x, y in zip(a, b))
                tests.append(dict(doc=label, baseline=base, k=k, llm_only=n_a, base_only=n_b,
                                  delta=(sum(a) - sum(b)) * 100 / res["n"], p=mcnemar_p(n_a, n_b)))
    holm(tests)

    other = next(n for n in REFERENCES.values() if n != reference)
    L = ["# Fair baselines: what does the model add?", "",
         f"Generated {datetime.now():%Y-%m-%d %H:%M} by `thesis/scripts/run_fair_baselines.py`. "
         f"Cached LLM arms: digest {', '.join(sorted({d for r in results.values() for d in r['llm_digests'] if d}))}; "
         f"rule-based arms built with code {_code_digest()}.", "",
         f"All baselines length-matched to `{reference}`, which is the tested arm; exact cosine search "
         f"over bge-m3; hit = 80 % anchor coverage. `{other}` is shown for reference only. "
         "`llm_incremental` carries the low-information filter and searches a smaller corpus.", ""]
    for label, res in results.items():
        L += [f"## {label}  (n = {res['n']}, target mean {res['target']} chars, cap {res['cap']})", "",
              "| Arm | Chunks | Mean | Median | Corpus | Hit@1 | Hit@3 | Hit@10 | MRR@10 | Ends mid-sentence | Anchors intact |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for name, r in res["rows"].items():
            L.append(f"| {'**' + name + '**' if name == reference else name} | {r['chunks']} | {r['mean_len']} "
                     f"| {r['median_len']} | {r['corpus_chars']:,} | {r['hit@1']} % | {r['hit@3']} % "
                     f"| {r['hit@10']} % | {r['mrr@10']:.3f} | {r['mid_sentence_pct']} % | {r['anchors_intact_pct']} % |")
        L.append("")
    L += [f"## Tests: `{reference}` against each baseline", "",
          f"Exact McNemar, Holm over m = {len(tests)} tests. Positive delta = LLM better.", "",
          "| Document | Baseline | k | delta | LLM only | baseline only | p | p (Holm) | Decision |",
          "|---|---|---:|---:|---:|---:|---:|---:|---|"]
    for t in sorted(tests, key=lambda t: (t["doc"], t["baseline"], t["k"])):
        L.append(f"| {t['doc']} | {t['baseline']} | {t['k']} | {t['delta']:+.1f} pp | {t['llm_only']} "
                 f"| {t['base_only']} | {t['p']:.4f} | {t['p_holm']:.4f} | "
                 f"{'**rejected**' if t['reject'] else 'not rejected'} |")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"\nWritten: {out}")
    return tests


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--docs", nargs="+", default=list(DOCS), choices=list(DOCS))
    ap.add_argument("--embedding-cache", type=Path, default=Path("eval_results/fair_baselines_embeddings.pkl"))
    ap.add_argument("--match", choices=list(REFERENCES), default="unfiltered",
                    help="which LLM arm's mean length the baselines are matched to, and tested against")
    args = ap.parse_args()

    suffix = "" if args.match == "unfiltered" else "_filtered"
    embed = CachedEmbedder(args.embedding_cache)
    results = {label: run_document(label, *DOCS[label], embed, args.match) for label in args.docs}
    tests = write_report(results, Path(f"thesis/results/FAIR_BASELINES{suffix}.md"), REFERENCES[args.match])
    # Chunks and top-10 rankings go along, so analyze_fair_arms.py can score
    # them in other ways without re-embedding.
    out_json = Path(f"eval_results/fair_baselines{suffix}.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(
        {"match": args.match, "reference": REFERENCES[args.match],
         "results": results, "tests": tests}, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
