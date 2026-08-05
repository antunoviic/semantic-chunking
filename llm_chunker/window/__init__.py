"""Sliding-window chunking strategy (the original / ablation approach)."""

from .detector import BoundaryDetector
from .prompt import BoundaryPrompt

__all__ = ["BoundaryDetector", "BoundaryPrompt"]
