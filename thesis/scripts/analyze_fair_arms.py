"""Scores the fair-baseline arms in ways the 80 % hit criterion cannot.

Reads what run_fair_baselines.py saved (chunks and top-10 rankings), so no
model and no embedder is needed. Everything here is post-hoc and secondary to
the pre-registered Hit@k.

  1. Token-level retrieval, after Smith & Troynikov (Chroma, 2024). Chunks and
     anchors are located in the whitespace-normalised document; per question
     the top-k chunks count together:
       recall     share of the anchor inside the union of retrieved chunks
       precision  anchor characters / retrieved characters
       IoU        anchor ∩ retrieved / anchor ∪ retrieved
     Recall no longer punishes an anchor split over two retrieved chunks, and
     precision charges for every character that comes along with the answer.
  2. Threshold sensitivity: Hit@k at 60 / 80 / 100 % coverage in one chunk, and
     at 80 % recall over the union of the top k.
  3. Structure: where the arms put boundaries relative to the document's
     numbered section headings, located at their position in the body text.
     check_heading_spans.py matches heading text anywhere, which also counts
     the table of contents, where every title recurs; positions do not.
     Overlapping splitters are scored again without overlap, since their
     overlap alone puts a heading inside the next chunk.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from app.document_reader import DocumentReader
from check_heading_spans import MIN_TITLE_CHARS, START_TOLERANCE
from eval.strategy_evaluator import StrategyEvaluator, _match_length, _recursive
from llm_semantic_chunker.incremental.heading_detection import extract_headings
from run_significance import mcnemar_p

KS = (1, 3)
THRESHOLDS = (0.6, 0.8, 1.0)
BOOTSTRAP = 2000
MIN_HEADINGS = 20      # below this the structure measure is reported as undefined


def norm(s: str) -> str:
    return " ".join(s.split())


class Locator:
    """Finds text in the normalised document, ignoring whitespace entirely.

    The chunker re-joins sentences with a space, also where the source had
    none ("B.C.;" becomes "B.C. ;"), so its chunks are not always substrings
    even after normalising whitespace. Matching on the text without any
    whitespace, then mapping back, locates them anyway.
    """

    def __init__(self, doc: str) -> None:
        self._pos = [i for i, ch in enumerate(doc) if not ch.isspace()]
        self._skeleton = "".join(doc[i] for i in self._pos)

    def find(self, text: str, cursor: int = 0) -> tuple[int, int, int] | None:
        """(start, end) in the document, plus the skeleton offset for the next search."""
        sk = "".join(text.split())
        if not sk:
            return None
        at = self._skeleton.find(sk, cursor)
        if at == -1:
            at = self._skeleton.find(sk)
        if at == -1:
            return None
        return self._pos[at], self._pos[at + len(sk) - 1] + 1, at

    def spans(self, chunks: list[str]) -> list[tuple[int, int] | None]:
        """Chunks come in document order, so each search starts after the last hit."""
        out: list[tuple[int, int] | None] = []
        cursor = 0
        for chunk in chunks:
            hit = self.find(chunk, cursor)
            out.append(hit[:2] if hit else None)
            if hit:
                cursor = hit[2] + 1
        return out


def union(spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    merged: list[list[int]] = []
    for a, b in sorted(spans):
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return [(a, b) for a, b in merged]


def token_scores(anchor: tuple[int, int], retrieved: list[tuple[int, int]]) -> tuple[float, float, float]:
    u = union(retrieved)
    size = sum(b - a for a, b in u)
    inter = sum(max(0, min(b, anchor[1]) - max(a, anchor[0])) for a, b in u)
    length = anchor[1] - anchor[0]
    return (inter / length,
            inter / size if size else 0.0,
            inter / (length + size - inter))


def body_headings(raw: str, ndoc: str) -> list[int]:
    """Positions in the normalised document of the numbered section headings."""
    out = []
    for pos, title in extract_headings(raw):
        if len(title) < MIN_TITLE_CHARS or not title[0].isdigit():
            continue
        before = norm(raw[:pos])
        npos = len(before) + (1 if before else 0)
        if ndoc[npos:npos + len(title)] == title:
            out.append(npos)
    return out


def structure(spans: list[tuple[int, int] | None], heads: list[int]) -> tuple[list[bool], int]:
    """Per heading whether it falls on a chunk boundary, and the chunks with a heading inside."""
    located = [s for s in spans if s]
    bounds = np.array(sorted({b for s in located for b in s}))
    on_boundary = [bool(np.min(np.abs(bounds - h)) <= START_TOLERANCE) for h in heads]
    mixing = sum(any(a + START_TOLERANCE < h < b - START_TOLERANCE for h in heads) for a, b in located)
    return on_boundary, mixing


def bootstrap_ci(diffs: np.ndarray) -> tuple[float, float]:
    rng = np.random.default_rng(0)
    means = rng.choice(diffs, size=(BOOTSTRAP, len(diffs)), replace=True).mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def analyse_document(label: str, res: dict, reference: str) -> dict:
    text = DocumentReader(res["document"]).extract_text()
    ndoc = norm(text)
    locate = Locator(ndoc)
    qa = json.loads(Path(res["questions"]).read_text(encoding="utf-8"))
    anchors = [hit[:2] if (hit := locate.find(pair["source_text"])) else None for pair in qa]
    usable = [i for i, a in enumerate(anchors) if a is not None]

    raw = (Path(res["document"]).read_text(encoding="utf-8")
           if res["document"].endswith(".txt") else text)
    heads = body_headings(raw, ndoc)

    arms = {}
    for name, chunks in res["chunks"].items():
        spans = locate.spans(chunks)
        top = res["top10"][name]
        per_q = {k: np.array([token_scores(anchors[i], [spans[j] for j in top[i][:k] if spans[j]])
                              for i in usable]) for k in KS}
        sens = {}
        for k in KS:
            cov = [[StrategyEvaluator.coverage(qa[i]["source_text"], chunks[j]) for j in top[i][:k]]
                   for i in range(len(qa))]
            for t in THRESHOLDS:
                sens[f"hit@{k}_{int(t * 100)}"] = sum(max(c) >= t - 1e-9 for c in cov) * 100 / len(qa)
            sens[f"union@{k}_80"] = float((per_q[k][:, 0] >= 0.8).sum() * 100 / len(usable))
        on_boundary, mixing = structure(spans, heads) if heads else ([], 0)
        arms[name] = {
            "unmapped_chunks": sum(s is None for s in spans),
            "per_q": per_q,
            "retrieved_chars": float(np.mean([sum(len(chunks[j]) for j in t[:3]) for t in top])),
            "sens": sens,
            "chunks": len(chunks),
            "on_boundary": on_boundary,
            "mixing": mixing,
        }
    extra = {}
    if len(heads) >= MIN_HEADINGS:
        for name, sentence_aware in (("recursive_matched", False), ("recursive_sent_matched", True)):
            chunks = _match_length(lambda size: _recursive(text, size, 0, sentence_aware=sentence_aware),
                                   res["target"], start=res["target"])
            on_boundary, mixing = structure(locate.spans(chunks), heads)
            extra[f"{name} (overlap 0)"] = {"chunks": len(chunks), "on_boundary": on_boundary, "mixing": mixing}
    ref_on = arms[reference]["on_boundary"]
    for r in [*arms.values(), *extra.values()]:
        b = sum(x and not y for x, y in zip(ref_on, r["on_boundary"]))
        c = sum(y and not x for x, y in zip(ref_on, r["on_boundary"]))
        r["boundary_p"] = mcnemar_p(b, c)
    return {"n": len(qa), "usable": len(usable), "headings": len(heads), "arms": arms,
            "structure_only": extra}


def write_report(data: dict, analysed: dict, out: Path) -> None:
    ref = data["reference"]
    L = ["# Fair-baseline arms: token-level retrieval, threshold sensitivity, structure", "",
         f"Generated {datetime.now():%Y-%m-%d %H:%M} by `thesis/scripts/analyze_fair_arms.py` from "
         f"the saved run (`--match {data['match']}`; tested arm `{ref}`). Post-hoc and secondary "
         "to the pre-registered Hit@k; no model or embedding calls.", ""]
    for label, a in analysed.items():
        L += [f"## {label}", "",
              f"{a['usable']} of {a['n']} anchors located in the normalised document.", "",
              "### Token-level retrieval (mean per question, %)", "",
              "| Arm | Recall@1 | Recall@3 | Precision@3 | IoU@3 | Chars retrieved@3 | Unmapped chunks |",
              "|---|---:|---:|---:|---:|---:|---:|"]
        for name, r in a["arms"].items():
            m1, m3 = r["per_q"][1].mean(axis=0) * 100, r["per_q"][3].mean(axis=0) * 100
            bold = "**" if name == ref else ""
            L.append(f"| {bold}{name}{bold} | {m1[0]:.1f} | {m3[0]:.1f} | {m3[1]:.2f} | {m3[2]:.2f} "
                     f"| {r['retrieved_chars']:,.0f} | {r['unmapped_chunks']} |")
        L += ["", f"Paired difference `{ref}` − baseline, mean over questions, bootstrap 95 % CI "
                  f"({BOOTSTRAP} resamples):", "",
              "| Baseline | Recall@3 (pp) | IoU@3 (pp) |", "|---|---:|---:|"]
        base_q = a["arms"][ref]["per_q"][3]
        for name, r in a["arms"].items():
            if name.startswith("llm_"):
                continue
            cells = []
            for col in (0, 2):
                d = (base_q[:, col] - r["per_q"][3][:, col]) * 100
                lo, hi = bootstrap_ci(d)
                cells.append(f"{d.mean():+.2f} [{lo:+.2f}, {hi:+.2f}]")
            L.append(f"| {name} | {cells[0]} | {cells[1]} |")
        L += ["", "### Hit@k under other criteria (%)", "",
              "| Arm | @1 60 % | @1 80 % | @1 100 % | @3 60 % | @3 80 % | @3 100 % | @3 union 80 % |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for name, r in a["arms"].items():
            s = r["sens"]
            L.append(f"| {name} | {s['hit@1_60']:.1f} | {s['hit@1_80']:.1f} | {s['hit@1_100']:.1f} "
                     f"| {s['hit@3_60']:.1f} | {s['hit@3_80']:.1f} | {s['hit@3_100']:.1f} | {s['union@3_80']:.1f} |")
        L += ["", "### Structure", ""]
        if a["headings"] < MIN_HEADINGS:
            L.append(f"Only {a['headings']} unambiguously numbered headings found — not defined on this document.")
        else:
            L += [f"{a['headings']} numbered section headings, at their position in the body text "
                  f"(tolerance {START_TOLERANCE} characters). *On a boundary* = a chunk starts or ends "
                  "at the heading. *Mixing chunks* = chunks with a heading strictly inside, i.e. the "
                  "tail of one section joined to the head of the next. The filtered arm dropped chunks, "
                  "and each gap counts as a boundary, so it is shown for reference only.", "",
                  f"The last column pairs each heading between `{ref}` and the arm (exact McNemar, "
                  "unadjusted).", "",
                  "| Arm | Chunks | Headings on a boundary | Mixing chunks | p vs tested arm |",
                  "|---|---:|---:|---:|---:|"]
            rows = {**a["arms"], **a["structure_only"]}
            for name, r in rows.items():
                on = sum(r["on_boundary"])
                p = "—" if name == ref else f"{r['boundary_p']:.2g}"
                L.append(f"| {name} | {r['chunks']} | {on} of {a['headings']} "
                         f"({on / a['headings'] * 100:.0f} %) "
                         f"| {r['mixing']} ({r['mixing'] / r['chunks'] * 100:.1f} %) | {p} |")
        L.append("")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"Written: {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--match", choices=("unfiltered", "filtered"), default="unfiltered")
    args = ap.parse_args()
    suffix = "" if args.match == "unfiltered" else "_filtered"
    data = json.loads(Path(f"eval_results/fair_baselines{suffix}.json").read_text(encoding="utf-8"))
    analysed = {}
    for label, res in data["results"].items():
        print(f"analysing {label}", flush=True)
        analysed[label] = analyse_document(label, res, data["reference"])
    write_report(data, analysed, Path(f"thesis/results/FAIR_ANALYSIS{suffix}.md"))


if __name__ == "__main__":
    main()
