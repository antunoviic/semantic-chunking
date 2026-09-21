from __future__ import annotations

import argparse
import re
from pathlib import Path

ROMAN = re.compile(r"^(?=[MDCLXVI]+$)M{0,4}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})$")
CREDIT = re.compile(r"^_?(Photo|From|By courtesy|In the|Nat\. Hist|Brit\. Mus|India Mus)", re.I)
MAP_LINE = re.compile(r"^Map:|^\s*Map\b.*_Map_")


def strip_gutenberg(text: str) -> str:
    """Drop the Gutenberg header and the licence footer."""
    start = text.find("*** START")
    if start >= 0:
        text = text[text.index("\n", start) + 1:]
    end = text.find("*** END")
    return text[:end] if end >= 0 else text


def body_only(text: str, first_chapter: str, tail_marker: str) -> str:
    #cut away front matter
    i = text.find(first_chapter)
    if i >= 0:
        text = text[i:]
    j = text.rfind(tail_marker)
    return text[:j] if j > len(text) // 2 else text


def drop_captions(text: str) -> str:
    #remove image captions and credits, keep chapter headings

    lines = text.split("\n")
    keep: list[str] = []
    prev_roman = False
    for raw in lines:
        s = raw.strip()
        if not s:
            keep.append("")
            prev_roman = False
            continue
        if ROMAN.match(s):
            keep.append(s)
            prev_roman = True
            continue
        if CREDIT.match(s) or MAP_LINE.match(s):
            prev_roman = False
            continue
        letters = [c for c in s if c.isalpha()]
        is_caps = bool(letters) and all(c.isupper() for c in letters)
        if is_caps and not prev_roman:
            continue                      # caption line
        keep.append(s)
        prev_roman = False
    return "\n".join(keep)


def unwrap(text: str) -> str:
    """Join hard-wrapped lines, keep blank lines as paragraph breaks."""
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


#requiring the title to be upper case in order to keep a paragraph
# starting with the pronoun "I" from being mistaken for a chapter
CHAPTER = re.compile(r"\n\n(?=[MDCLXVI]+ [A-Z][A-Z ,'\-.]{4,}\n)")


def truncate_at_chapter(text: str, limit: int) -> tuple[str, int]:
    """Cut at the last chapter boundary before `limit` — never mid-chapter."""
    cuts = [m.start() for m in CHAPTER.finditer(text)]
    if len(text) <= limit:
        return text, 0
    usable = [c for c in cuts if c <= limit]
    if not usable:
        return text[:limit], 0
    return text[:usable[-1]], len(cuts) - len(usable)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("source")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-chars", type=int, default=250_000)
    ap.add_argument("--first-chapter", default="\nI\nTHE WORLD IN SPACE")
    ap.add_argument("--tail-marker", default="CHRONOLOGICAL TABLE")
    args = ap.parse_args()

    raw = Path(args.source).expanduser().read_text(encoding="utf-8")
    print(f"Source ............ {len(raw):,} chars")

    t = strip_gutenberg(raw)
    print(f"after boilerplate . {len(t):,}")
    t = body_only(t, args.first_chapter, args.tail_marker)
    print(f"after front/back .. {len(t):,}")
    t = drop_captions(t)
    print(f"after captions .... {len(t):,}")
    t = unwrap(t)
    print(f"after unwrapping .. {len(t):,}")
    t, dropped = truncate_at_chapter(t, args.max_chars)
    print(f"after truncation .. {len(t):,}  ({dropped} chapters left out)")

    chapters = len(CHAPTER.findall("\n\n" + t))
    print(f"chapters kept ..... {chapters}")

    out = Path(args.out).expanduser()
    out.write_text(t + "\n", encoding="utf-8")
    print(f"\nWritten: {out}")
    print(f"Next:  python tools/check_document.py {out}")


if __name__ == "__main__":
    main()
