"""Incremental (sequential) chunking strategy — the default / primary method."""

from .detector import IncrementalBoundaryDetector
from .prompt import IncrementalBoundaryPrompt

__all__ = ["IncrementalBoundaryDetector", "IncrementalBoundaryPrompt"]
