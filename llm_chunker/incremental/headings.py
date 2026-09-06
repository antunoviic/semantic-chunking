from __future__ import annotations

import re

from .detector import IncrementalBoundaryDetector

HEADING_RE = re.compile(
    r"^\s*(?:"
    r"\d+(?:\.\d+)+\.?\s+\S"                          # multi-level: 3.2 X
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
    #Group sentences into sections, heading-like sentence starts new one
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

    def __init__(self, *args, heading_re: re.Pattern[str] = HEADING_RE, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.heading_re = heading_re

    def detect_and_assemble(self, sentences: list[str], raw_text: str | None = None) -> list[str]:
        """`raw_text` wird ignoriert — diese Klasse matcht auf Saetzen, siehe
        Modul-Docstring in heading_detection.py fuer die staerkere Alternative."""
        self.reset_stats()
        chunks: list[str] = []
        segments = split_at_headings(sentences, self.heading_re, self.verbose)
        for segment in segments:
            chunks.extend(self._run(segment))
        self.boundary_stats["heading"] = max(0, len(segments) - 1)
        return chunks
