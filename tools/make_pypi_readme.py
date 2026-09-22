"""Build the PyPI long description from README.md.

The project page on PyPI is the package's documentation. The evaluation harness
behind the thesis only means something to someone who has the repository — the
documents and question sets are not in the distribution — so that chapter is cut
and replaced by a pointer. GitHub keeps the full text.

Both are generated from one source, so they cannot drift apart. Without this,
the page goes stale silently: setuptools does not fail when the readme file is
missing or outdated, it just ships whatever it finds.

    python tools/make_pypi_readme.py          # rewrite README-pypi.md
    python tools/make_pypi_readme.py --check  # exit 1 if it is out of date

Run --check before building a release.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

SOURCE = Path("README.md")
TARGET = Path("README-pypi.md")

# The chapter to drop, by its heading. Cutting by heading rather than by marker
# comments keeps README.md free of anything a reader would see in the source.
CUT_HEADING = "## Evaluation harness"

POINTER = """The repository additionally holds the evaluation harness behind the bachelor
thesis this library was written for, together with the documents and question
sets it was measured on — see the
[README on GitHub](https://github.com/antunoviic/semantic-chunking#evaluation-harness).

---

"""


def _section_bounds(lines: list[str], heading: str) -> tuple[int, int]:
    """Line range of the chapter: its heading up to the next one of equal rank.

    Fenced code blocks are skipped, so a `## ...` inside an example does not end
    the chapter early.
    """
    rank = len(heading) - len(heading.lstrip("#"))
    start = next((i for i, l in enumerate(lines) if l.rstrip() == heading), -1)
    if start < 0:
        raise SystemExit(f"{SOURCE}: heading {heading!r} not found — "
                         f"rename it here too, or the page would keep the chapter.")

    fence = False
    for i in range(start + 1, len(lines)):
        if lines[i].lstrip().startswith("```"):
            fence = not fence
            continue
        if fence:
            continue
        stripped = lines[i].rstrip()
        if stripped.startswith("#"):
            if len(stripped) - len(stripped.lstrip("#")) <= rank:
                return start, i
    return start, len(lines)


def build(text: str) -> str:
    lines = text.splitlines(keepends=True)
    start, end = _section_bounds(lines, CUT_HEADING)
    head = "".join(lines[:start])
    tail = "".join(lines[end:])
    return (head + POINTER + tail).rstrip() + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="exit 1 if README-pypi.md does not match README.md")
    args = ap.parse_args()

    source = SOURCE.read_text(encoding="utf-8")
    want = build(source)
    have = TARGET.read_text(encoding="utf-8") if TARGET.exists() else None

    if args.check:
        if want != have:
            print(f"{TARGET} is out of date — run: python {sys.argv[0]}")
            raise SystemExit(1)
        print(f"{TARGET} is up to date")
        return

    TARGET.write_text(want, encoding="utf-8")
    print(f"Wrote {TARGET}: {len(want)} characters, "
          f"{len(source) - len(want)} fewer than {SOURCE}")


if __name__ == "__main__":
    main()
