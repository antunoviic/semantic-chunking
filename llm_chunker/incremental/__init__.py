"""Incremental (sequential) chunking strategy — the default / primary method."""

from .detector import IncrementalBoundaryDetector
from .headings import HeadingAwareBoundaryDetector, split_at_headings, HEADING_RE
from .heading_detection import (HeadingOnlyPrompt, extract_headings,
                                heading_sentence_indices,
                                looks_like_heading_candidate)
from .hybrid import HybridHeadingBoundaryDetector
from .prompt import IncrementalBoundaryPrompt, SplitPointPrompt

__all__ = [
    "IncrementalBoundaryDetector",
    "HeadingAwareBoundaryDetector",
    "HybridHeadingBoundaryDetector",
    "split_at_headings",
    "HEADING_RE",
    "HeadingOnlyPrompt",
    "looks_like_heading_candidate",
    "extract_headings",
    "heading_sentence_indices",
    "IncrementalBoundaryPrompt",
    "SplitPointPrompt",
]
