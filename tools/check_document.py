from __future__ import annotations

import re
import sys
from pathlib import Path
from pathlib import Path as _Path

sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))   # project root importable

from app.document_reader import DocumentReader
from llm_chunker.incremental.headings import HEADING_RE
from llm_chunker.text_splitter import TextSplitter

STEP_SENTENCES = 2          # as in the ablation setup
SECONDS_PER_CALL = 5.0      


def check(path: str) -> None:
    print("=" * 74)
    print(f"  {Path(path).name}")
    print("=" * 74)

    text = DocumentReader(path).extract_text()
    sents = TextSplitter(sentences_per_chunk=3).split_sentences(text)

    words = text.split()
    avg_word = sum(len(w) for w in words) / max(1, len(words))
    avg_sent = sum(len(s) for s in sents) / max(1, len(sents))
    # Two-column PDFs produce very short sentences and mangled words
    tiny = sum(1 for s in sents if len(s) < 25)
    weird = sum(1 for w in words if len(w) > 30)

    print("\n[1] Extraction")
    print(f"    characters ......... {len(text):,}")
    print(f"    words .............. {len(words):,}   (avg {avg_word:.1f} chars)")
    print(f"    sentences .......... {len(sents):,}   (avg {avg_sent:.0f} chars)")
    print(f"    very short sents ... {tiny:,} ({tiny*100//max(1,len(sents))} %)"
          f"  <- high = possibly two-column")
    print(f"    monster words >30 .. {weird:,}                <- high = extraction mangled")
    verdict = "OK" if avg_word < 9 and tiny * 100 / max(1, len(sents)) < 35 else "SUSPICIOUS"
    print(f"    -> extraction: {verdict}")

    print("\n    Text sample (characters 3000-3400):")
    print(f"      {text[3000:3400]!r}"[:400])

    # heading detection
    matches = [s.strip() for s in sents if HEADING_RE.match(s)]
    numbered = [s for s in matches if re.match(r"^\d", s)]
    named = [s for s in matches if not re.match(r"^\d", s)]

    # Quality of the hits rather than their count:
    toc = [s for s in matches if "...." in s or "…" in s]
    tabular = [s for s in numbered if re.match(r"^\d+[.,]\d+\s+\d", s)]
    good = [s for s in matches if s not in toc and s not in tabular]

    # The actual failure mode: NLTK splits the section number into its own
    orphan = [s for s in sents if re.fullmatch(r"\s*\d+(?:\.\d+)*\.?\s*", s)]

    print("\n[2] Headings")
    print(f"    regex hits ........... {len(matches)}   "
          f"(numbered {len(numbered)}, whitelist {len(named)})")
    print(f"      of which TOC ....... {len(toc)}   (dot leaders, worthless)")
    print(f"      of which tabular ... {len(tabular)}   "
          f"(FALSE POSITIVE: cuts into data tables)")
    print(f"      usable ............. {len(good)}")
    print(f"    orphaned numbers ..... {len(orphan)}   "
          f"<- split off as their own sentence, undetectable")

    if orphan and len(orphan) > len(good):
        print("    -> WARNING: more orphaned section numbers than usable hits")
        print("       (the RFC/EFSA failure mode: NLTK splits the number off)")
    if tabular and len(tabular) > len(good) / 2:
        print("    -> WARNING: many tabular false positives — a heading split would hurt")
    if good and not orphan and not tabular:
        print("    -> heading detection works cleanly")

    print("    Usable hits (excerpt):")
    for s in (good or matches)[:5]:
        print(f"       {s[:66]!r}")
    if tabular:
        print("    False positives (excerpt):")
        for s in tabular[:3]:
            print(f"       {s[:66]!r}")

    # estimations about running
    calls = len(sents) // STEP_SENTENCES
    hours = calls * SECONDS_PER_CALL / 3600
    print("\n[3] Effort (chunking)")
    print(f"    estimated LLM calls .. ~{calls:,}")
    print(f"    estimated runtime .... ~{hours:.1f} h  (at {SECONDS_PER_CALL:.0f} s/call)")
    print("    Measured for comparison: nasa 4,846 sentences -> ~2,400 calls -> 4.5 h;")
    print("    rfc9110 3,882 sentences -> ~1,940 calls -> 25.5 h under heavy swapping.")
    print("    Memory pressure dominates this estimate by an order of magnitude.")
    print()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    for p in sys.argv[1:]:
        if not Path(p).exists():
            print(f"!! file missing: {p}")
            continue
        check(p)
