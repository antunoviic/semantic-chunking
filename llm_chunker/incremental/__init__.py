"""Incremental (sequential) chunking strategy — the default / primary method."""

from .detector import IncrementalBoundaryDetector
from .headings import HeadingAwareBoundaryDetector, split_at_headings, HEADING_RE
from .prompt import IncrementalBoundaryPrompt, SplitPointPrompt

__all__ = [
    "IncrementalBoundaryDetector",
    "HeadingAwareBoundaryDetector",
    "split_at_headings",
    "HEADING_RE",
    "IncrementalBoundaryPrompt",
    "SplitPointPrompt",
]
