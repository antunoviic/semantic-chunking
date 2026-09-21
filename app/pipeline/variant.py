"""Names the ablation arms.

Each arm needs two names: one for its cache file and one for the results table.
Both are derived entirely from the `ChunkerConfig`. An earlier version repeated
the same four settings here, so adding a detection mode meant editing four
places.

    cache_key(cfg)                      -> "incremental_headings_hybrid"
    cache_key(cfg, heading_mode="lines")-> "incremental_headings_lines"
    cache_key(cfg, enrich=True)         -> "incremental_headings_hybrid_enriched"
    label(cfg)                          -> "llm_incremental_headings_hybrid"
"""
from __future__ import annotations

import dataclasses

from llm_semantic_chunker import ChunkerConfig

# The originally shipped mode gets no suffix: the arm is called
# "incremental_headings", not "incremental_headings_regex".
_DEFAULT_HEADING_MODE = "regex"


def _with(cfg: ChunkerConfig, **overrides) -> ChunkerConfig:
    """The same configuration with individual values replaced.

    Lets the collector ask for counterparts without building a second
    configuration by hand: `cache_key(cfg, respect_headings=False)`.
    """
    return dataclasses.replace(cfg, **overrides) if overrides else cfg


def _suffix(cfg: ChunkerConfig) -> str:
    if cfg.mode == "window":
        base = ""                                   # window runs under the plain document name
    else:
        base = "incremental_headings" if cfg.respect_headings else "incremental"
        if cfg.respect_headings and cfg.heading_mode != _DEFAULT_HEADING_MODE:
            base += f"_{cfg.heading_mode}"
        if not cfg.smart_split:
            base += "_midpoint"
        if not cfg.filter_low_info:
            # Its own key: otherwise the filter ablation would overwrite the
            # main cache, and their comparison is what separates the filter
            # effect from the boundary effect.
            base += "_nofilter"
    if cfg.enrich:
        # With enrich=True the chunker prefixes every chunk with
        # "[Topic: ...]". That is different text and must never be stored
        # under the name of the unenriched arm.
        base = f"{base}_enriched" if base else "enriched"
    return base


def cache_key(cfg: ChunkerConfig, **overrides) -> str:
    """Cache file name for this arm, without document stem and extension."""
    return _suffix(_with(cfg, **overrides))


def label(cfg: ChunkerConfig, **overrides) -> str:
    """Name of this arm in the results table."""
    cfg = _with(cfg, **overrides)
    if cfg.mode == "window":
        return "llm_window" + ("_enriched" if cfg.enrich else "")
    return "llm_" + _suffix(cfg)
