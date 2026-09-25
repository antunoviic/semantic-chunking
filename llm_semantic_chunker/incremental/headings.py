from __future__ import annotations

import re

from .detector import IncrementalBoundaryDetector
from .._logging import get_logger

logger = get_logger(__name__)

HEADING_RE = re.compile(
    r"^\s*(?:"
    r"\d+(?:\.\d+)+\.?\s+\S"                          # multi-level: 3.2 X
    r"|\d+\.\s+[A-Z]"                                 # top-level:   1. Introduction
    r"|(?:Abstract|Introduction|Background|Related Work|Methodology|Methods|Materials|"
    r"Results|Discussion|Conclusion|Conclusions|References|Bibliography|"
    r"Acknowledgements|Acknowledgments|Appendix|Key Points|Summary|Overview)\b"
    r")"
)


_MAX_HEADING_LEN = 90


def is_heading_sentence(sentence: str, heading_re: re.Pattern[str] = HEADING_RE) -> bool:
    s = sentence.strip()
    if not s or len(s) > _MAX_HEADING_LEN:
        return False
    if not heading_re.match(s):
        return False
    if s.endswith((".", "!", "?", ":", ";", ",")) and not re.match(r"^\s*\d", s):
        return False
    return True


def split_at_headings(
    sentences: list[str],
    heading_re: re.Pattern[str] = HEADING_RE,
) -> list[list[str]]:
    #Group sentences into sections, heading-like sentence starts new one
    segments: list[list[str]] = []
    current: list[str] = []
    for sentence in sentences:
        if current and is_heading_sentence(sentence, heading_re):
            logger.debug(f"[headings] boundary -> {sentence[:60]!r}")
            segments.append(current)
            current = [sentence]
        else:
            current.append(sentence)
    if current:
        segments.append(current)
    return segments


class HeadingAwareBoundaryDetector(IncrementalBoundaryDetector):
    """Forces a boundary at headings found by a sentence-level pattern.

    The sentence list is pre-segmented at each heading *before* the loop runs,
    so the boundary sits exactly at the heading sentence. The cost is that the
    pattern only sees text that already survived sentence tokenisation, where
    a heading on its own line may have been merged into the sentence after it.
    See `HybridHeadingBoundaryDetector` for the line-based alternative.
    """

    def __init__(self, *args, heading_re: re.Pattern[str] = HEADING_RE, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.heading_re = heading_re

    def detect_and_assemble(self, sentences: list[str], raw_text: str | None = None) -> list[str]:
        """`raw_text` is ignored: this detector matches on sentences and splits
        exactly at the heading sentence. The line-based detection on the raw text
        lives in heading_detection.py and is used by HybridHeadingBoundaryDetector."""
        self.reset_stats()
        chunks: list[str] = []
        segments = split_at_headings(sentences, self.heading_re)
        for segment in segments:
            chunks.extend(self._run(segment))
        heading_boundaries = max(0, len(segments) - 1)
        self.boundary_stats["heading"] = heading_boundaries
        self.boundary_stats["end"] = max(0, self.boundary_stats["end"] - heading_boundaries)
        return chunks
