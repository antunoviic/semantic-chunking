"""Heading-aware chunking: a structural guardrail on top of the incremental detector.

Addresses the observation that a purely semantic (LLM YES/NO) boundary decision can
merge across a document's structure, producing chunks that span several chapters
(e.g. Abstract + Introduction + Key Points in one chunk).

The guardrail is deliberately kept OUT of IncrementalBoundaryDetector: that class
stays responsible for the semantic decision and the size-cap split only. Here we
add hard boundaries at detected headings and chunk each section on its own.

Scope / known limits (heuristic, not a full document parser):
  * covers numbered sections (3.2 X, 15.5.1 X, 1. Introduction) and a whitelist of
    common paper sections;
  * does NOT cover ALL-CAPS titles, markdown '#', 'Chapter N', roman numerals, or
    unnumbered headings in prose books;
  * a missed heading is harmless (falls back to pure LLM chunking), whereas a false
    positive (e.g. a sentence starting "3.2 million people ...") over-splits.
"""

from __future__ import annotations

import re

from .detector import IncrementalBoundaryDetector

# A sentence starting like this marks a new section/heading -> a chunk must never
# span across it.
HEADING_RE = re.compile(
    r"^\s*(?:"
    r"\d+(?:\.\d+)+\.?\s+\S"                          # multi-level: 3.2 X / 6.0 X / 15.5.1 X
    r"|\d+\.\s+[A-Z]"                                 # top-level:   1. Introduction
    r"|(?:Abstract|Introduction|Background|Related Work|Methodology|Methods|Materials|"
    r"Results|Discussion|Conclusion|Conclusions|References|Bibliography|"
    r"Acknowledgements|Acknowledgments|Appendix|Key Points|Summary|Overview)\b"
    r")"
)


def split_at_headings(
    sentences: list[str],
    heading_re: re.Pattern[str] = HEADING_RE,
    verbose: bool = False,
) -> list[list[str]]:
    """Group sentences into sections; a heading-like sentence starts a new one."""
    segments: list[list[str]] = []
    current: list[str] = []
    for sentence in sentences:
        if current and heading_re.match(sentence):
            if verbose:
                print(f"[headings] boundary -> {sentence[:60]!r}")
            segments.append(current)
            current = [sentence]
        else:
            current.append(sentence)
    if current:
        segments.append(current)
    return segments


class HeadingAwareBoundaryDetector(IncrementalBoundaryDetector):
    """IncrementalBoundaryDetector that never lets a chunk span a heading.

    Sentences are first cut into sections at detected headings (hard boundaries);
    each section is then chunked by the inherited incremental logic (LLM YES/NO
    plus size-cap split). Since this can only ADD boundaries, never remove them,
    the worst case is a slightly smaller chunk — never a merged chapter.

    Use plain IncrementalBoundaryDetector (CLI: --no-headings) for the ablation.
    """

    def __init__(self, *args, heading_re: re.Pattern[str] = HEADING_RE, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.heading_re = heading_re

    def detect_and_assemble(self, sentences: list[str]) -> list[str]:
        self.reset_stats()
        chunks: list[str] = []
        segments = split_at_headings(sentences, self.heading_re, self.verbose)
        for segment in segments:
            # _run statt detect_and_assemble: sonst wuerden die Zaehler je
            # Abschnitt zurueckgesetzt.
            chunks.extend(self._run(segment))
        # Jede Segmentgrenze ausser der letzten ist eine erzwungene Heading-Grenze
        self.boundary_stats["heading"] = max(0, len(segments) - 1)
        return chunks
