from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from pathlib import Path as _Path

sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))   # project root importable

from app.document_reader import DocumentReader

MIN_LEN = 40
CRITERIA = ("groundedness", "relevance", "standalone")


def norm(s: str) -> str:
    return " ".join(s.split())


def skeleton(s: str) -> tuple[str, list[int]]:
    #ASCII skeleton plus a mapping back to original positions.
    out, pos = [], []
    for i, ch in enumerate(s):
        if ch.isascii() and ch.isalnum():
            out.append(ch.lower())
            pos.append(i)
    return "".join(out), pos


def load_json_items(path: Path) -> list[dict]:
    #Reads collected LLM replies
    raw = path.read_text(encoding="utf-8")
    try:
        data = json.loads(raw)
        return data if isinstance(data, list) else [data]
    except json.JSONDecodeError:
        pass
    items: list[dict] = []
    for m in re.finditer(r"\[\s*\{.*?\}\s*\]", raw, re.DOTALL):
        try:
            items.extend(json.loads(m.group()))
        except json.JSONDecodeError:
            continue
    if not items:
        raise SystemExit(f"No JSON objects found in {path}.")
    print(f"[parse] read {len(items)} entries from multiple JSON blocks")
    return items


def load_ratings(path: Path, threshold: int) -> tuple[set[str], dict]:
    #Returns the questions that pass all criteria, plus a per-criterion tally.

    items = load_json_items(path)
    passing: set[str] = set()
    failed = {c: 0 for c in CRITERIA}
    for it in items:
        q = norm(it.get("question", ""))
        if not q:
            continue
        below = [c for c in CRITERIA if int(it.get(c, 0)) < threshold]
        for c in below:
            failed[c] += 1
        if not below:
            passing.add(q)
    return passing, failed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("document")
    ap.add_argument("candidates", help="collected LLM replies (JSON)")
    ap.add_argument("--merge", default=None, help="append an existing set")
    ap.add_argument("--ratings", default=None,
                    help="critique replies with groundedness/relevance/standalone")
    ap.add_argument("--min-rating", type=int, default=4,
                    help="minimum score on every criterion (default 4 of 5)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    text = DocumentReader(args.document).extract_text()
    text_n = norm(text)
    text_sk, text_pos = skeleton(text)

    cands = load_json_items(Path(args.candidates))
    if args.merge and Path(args.merge).exists():
        old = json.loads(Path(args.merge).read_text(encoding="utf-8"))
        print(f"[merge] carrying over {len(old)} existing questions")
        cands = old + cands

    passing: set[str] | None = None
    rating_failures: dict = {}
    if args.ratings:
        passing, rating_failures = load_ratings(Path(args.ratings), args.min_rating)
        print(f"[critique] {len(passing)} questions scored >= {args.min_rating} "
              f"on all three criteria")

    ok: list[dict] = []
    repaired = dropped_short = dropped_dup = dropped_ambig = unresolved = 0
    dropped_rating = 0
    seen: set[str] = set()
    problems: list[str] = []

    for q in cands:
        question = (q.get("question") or "").strip()
        src = (q.get("source_text") or "").strip()
        if not question or not src:
            unresolved += 1
            continue

        if passing is not None and norm(question) not in passing:
            dropped_rating += 1
            continue

        if norm(src) in text_n:
            fixed = src
        else:
            # skeleton repair: slice the original text at the located position
            sk, _ = skeleton(src)
            idx = text_sk.find(sk) if sk else -1
            if idx == -1 or not sk:
                unresolved += 1
                problems.append(f"not locatable: {src[:70]!r}")
                continue
            start = text_pos[idx]
            end = text_pos[idx + len(sk) - 1] + 1
            fixed = text[start:end]
            repaired += 1

        if len(fixed) < MIN_LEN:
            dropped_short += 1
            continue
        key = norm(fixed)
        if key in seen:
            dropped_dup += 1
            continue
        if text_n.count(key) > 1:          # occurs more than once
            dropped_ambig += 1
            continue
        seen.add(key)
        ok.append({"question": question, "source_text": fixed})

    stem = Path(args.document).stem
    out = Path(args.out or f"eval_cache/{stem}_questions_verified.json")
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(ok, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 64)
    print(f"  candidates .............. {len(cands)}")
    print(f"  repaired (skeleton) ..... {repaired}")
    if args.ratings:
        print(f"  dropped: low rating ..... {dropped_rating}")
        for c in CRITERIA:
            print(f"      below on {c:<13} {rating_failures.get(c, 0)}")
    print(f"  dropped: too short ...... {dropped_short}")
    print(f"  dropped: duplicate ...... {dropped_dup}")
    print(f"  dropped: ambiguous ...... {dropped_ambig}")
    print(f"  dropped: not locatable .. {unresolved}")
    print(f"  -> USABLE ............... {len(ok)}")
    print("=" * 64)
    print(f"Written: {out}")
    if problems:
        print("\nUnresolvable (excerpt) — usually invented or heavily reworded passages:")
        for p in problems[:8]:
            print(f"  {p}")
    if len(ok) < 30:
        print("\nWARNING: fewer than 30 questions — too few for meaningful metrics.")


if __name__ == "__main__":
    main()
