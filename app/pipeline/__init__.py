"""Chunking- und Auswertungspipeline.

    runner.py      Orchestrierung eines Laufs
    strategies.py  welche Chunk-Mengen verglichen werden
    variant.py     Cache-Schluessel und Report-Namen je Ablationsarm
"""
from .runner import Pipeline
from .variant import ChunkVariant

__all__ = ["Pipeline", "ChunkVariant"]
