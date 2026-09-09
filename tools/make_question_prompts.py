from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from pathlib import Path as _Path

sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))   # project root importable

from app.document_reader import DocumentReader

OUT_ROOT = Path("question_prompts")

STYLE_HINT = {
    "literal": (
        "Phrase the question CLOSE to the wording of the text (high lexical "
        "overlap). Use the terminology of the document."
    ),
    "natural": (
        "Phrase the question the way a real user would ask it: conversational, "
        "PARAPHRASED, with as LITTLE verbatim overlap with the text as possible. "
        "Avoid the document's technical terms wherever an everyday wording exists."
    ),
}

TEMPLATE = """You are helping to build an evaluation dataset for a RAG system.

Below is section {idx} of {n_sections} of a document. Produce EXACTLY
{per_section} question/answer pairs from it.

RULES — the second one matters most:
1. GROUNDEDNESS. Every question must be answerable from this section alone,
   unambiguously and without outside knowledge.
2. "source_text" must be copied CHARACTER FOR CHARACTER from the section —
   literally a substring. Do not rephrase, shorten, correct or substitute any
   character (including quotation marks and dashes). A single altered character
   makes the pair unusable.
3. "source_text" is 60-300 characters long and contains the complete answer.
4. RELEVANCE. Ask about concrete facts (numbers, definitions, conditions,
   procedures) that a real user would plausibly look up — not about opinions or
   summaries.
5. STAND-ALONE. The question must make sense on its own. Someone who cannot see
   this section must still understand what is being asked. No "in this section",
   no "the above mentioned", no unresolved pronouns.
6. The answer must NOT appear in several places of the document — otherwise a
   materially correct retrieval counts as a miss. Pick unique passages.
7. No questions about headings, table of contents, bibliography, author lists,
   table headers or page numbers.
8. {style_hint}
9. Spread the questions across the WHOLE section, not just its beginning.

Reply with a JSON array ONLY, no commentary before or after:
[{{"question": "...", "source_text": "..."}}]
{avoid_block}
--- SECTION {idx}/{n_sections} ---
{section_text}
--- END OF SECTION ---
"""

CRITIQUE_TEMPLATE = """You are reviewing an evaluation dataset for a RAG system.

Rate EVERY question below on three criteria, each from 1 to 5:

  groundedness  Can it be answered from its own source_text, unambiguously and
                without outside knowledge?  (1 = not at all, 5 = fully)
  relevance     Would a real user of this document plausibly ask it?
                (1 = artificial, 5 = clearly useful)
  standalone    Is it understandable without seeing the surrounding section?
                (1 = depends on context, 5 = fully self-contained)

Judge each question on its own. Do not rewrite anything and do not add or drop
entries — return exactly as many objects as you receive, in the same order.

Reply with a JSON array ONLY:
[{{"question": "...", "source_text": "...", "groundedness": N,
   "relevance": N, "standalone": N}}]

--- QUESTIONS ---
<paste the collected JSON here>
--- END ---
"""


def split_sections(text: str, n: int) -> list[str]:
    """Splits at paragraph boundaries into roughly equal sections (never mid-word)."""
    paras = text.split("\n\n")
    target = max(1, len(text) // n)
    sections, cur, size = [], [], 0
    for p in paras:
        cur.append(p)
        size += len(p) + 2
        if size >= target and len(sections) < n - 1:
            sections.append("\n\n".join(cur))
            cur, size = [], 0
    if cur:
        sections.append("\n\n".join(cur))
    return [s for s in sections if s.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("document")
    ap.add_argument("--total", type=int, default=100)
    ap.add_argument("--sections", type=int, default=20)
    ap.add_argument("--style", choices=["literal", "natural"], default="literal")
    ap.add_argument("--avoid", default=None, help="existing question set (JSON)")
    ap.add_argument("--tag", default=None,
                    help="suffix for the output directory. Use it when growing an "
                         "existing set: without it the prompts that produced the "
                         "current questions would be overwritten and the provenance "
                         "of the benchmark lost.")
    ap.add_argument("--out-root", default=str(OUT_ROOT),
                    help="target directory (default question_prompts). Use "
                         "expansion_prompts when growing a set, so the files still "
                         "to be processed cannot be confused with the ones already "
                         "answered.")
    ap.add_argument("--critique", action="store_true",
                    help="also write the second-stage rating prompt "
                         "(groundedness / relevance / stand-alone)")
    ap.add_argument("--excerpt-chars", type=int, default=15000,
                    help="max characters per section in the prompt (default 15000). "
                         "Longer sections are trimmed to an excerpt from the middle, "
                         "otherwise the prompts become unwieldy to paste.")
    args = ap.parse_args()

    stem = Path(args.document).stem
    text = DocumentReader(args.document).extract_text()
    sections = split_sections(text, args.sections)
    per_section = max(1, math.ceil(args.total / len(sections)))

    # Trim long sections to an excerpt
    truncated = 0
    trimmed = []
    for sec in sections:
        if len(sec) > args.excerpt_chars:
            start = (len(sec) - args.excerpt_chars) // 2
            # align to a paragraph boundary so no sentence starts mid-way
            cut = sec.find("\n\n", start)
            start = cut + 2 if 0 <= cut < start + 2000 else start
            sec = sec[start:start + args.excerpt_chars]
            truncated += 1
        trimmed.append(sec)
    sections = trimmed

    # Exclude existing source_texts 
    avoid_block = ""
    if args.avoid and Path(args.avoid).exists():
        old = json.loads(Path(args.avoid).read_text(encoding="utf-8"))
        snippets = [q["source_text"][:80] for q in old]
        avoid_block = (
            f"\nALREADY USED — pick DIFFERENT passages ({len(snippets)} existing, "
            f"showing {min(60, len(snippets))}):\n"
            + "\n".join(f"  - {s}..." for s in snippets[:60]) + "\n"
        )

    suffix = f"_{args.tag}" if args.tag else ""
    out_dir = Path(args.out_root) / f"{stem}_{args.style}{suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)
    for f in out_dir.glob("section_*.txt"):
        f.unlink()

    for i, sec in enumerate(sections, 1):
        prompt = TEMPLATE.format(
            idx=i, n_sections=len(sections), per_section=per_section,
            style_hint=STYLE_HINT[args.style], avoid_block=avoid_block,
            section_text=sec,
        )
        # Document name in every filename
        #cause pasted into external LLM
        (out_dir / f"section_{i:02d}_{stem}.txt").write_text(prompt, encoding="utf-8")

    if args.critique:
        (out_dir / f"critique_{stem}.txt").write_text(CRITIQUE_TEMPLATE, encoding="utf-8")

    covered = sum(len(s) for s in sections)
    print(f"Document .......... {stem}  ({len(text):,} chars)")
    print(f"Sections .......... {len(sections)}  (avg {covered//len(sections):,} chars per prompt)")
    if truncated:
        print(f"  trimmed ......... {truncated} to {args.excerpt_chars:,} chars each "
              f"-> {covered*100//len(text)} % of the document covered")
    print(f"Questions/section . {per_section}  ->  ~{per_section*len(sections)} total")
    print(f"Style ............. {args.style}")
    print(f"\nPrompts written to: {out_dir}/")
    print("\nNext steps:")
    print(f"  1. Paste the files in {out_dir}/ into a strong external LLM, one at a time")
    print("  2. Collect all JSON replies into ONE file (arrays back to back is fine)")
    if args.critique:
        print(f"  3. Paste {out_dir}/critique_prompt.txt with that file, keep only "
              f"questions rated >= 4 on all three criteria")
        print(f"  4. python tools/verify_questions.py {args.document} <collected.json>")
    else:
        print(f"  3. python tools/verify_questions.py {args.document} <collected.json>")


if __name__ == "__main__":
    main()
