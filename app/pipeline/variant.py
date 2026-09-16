"""Benennt die Ablationsarme.

Jeder Arm braucht zwei Namen: einen fuer die Cache-Datei und einen fuer die
Ergebnistabelle. Beide leiten sich vollstaendig aus der `ChunkerConfig` ab —
frueher hielt diese Datei dieselben vier Einstellungen noch einmal selbst, und
eine neue Erkennungsart musste an vier Stellen nachgetragen werden.

    cache_key(cfg)                      -> "incremental_headings_hybrid"
    cache_key(cfg, heading_mode="lines")-> "incremental_headings_lines"
    label(cfg)                          -> "llm_incremental_headings_hybrid"
"""
from __future__ import annotations

import dataclasses

from llm_semantic_chunker import ChunkerConfig

# Der urspruenglich ausgelieferte Modus bekommt kein Suffix: der Arm heisst
# "incremental_headings", nicht "incremental_headings_regex".
_DEFAULT_HEADING_MODE = "regex"


def _with(cfg: ChunkerConfig, **overrides) -> ChunkerConfig:
    """Dieselbe Konfiguration mit einzelnen geaenderten Werten.

    Damit fragt der Sammler nach Gegenstuecken, ohne eine zweite Konfiguration
    von Hand bauen zu muessen: `cache_key(cfg, respect_headings=False)`.
    """
    return dataclasses.replace(cfg, **overrides) if overrides else cfg


def _suffix(cfg: ChunkerConfig) -> str:
    if cfg.mode == "window":
        return ""                                   # window laeuft unter dem Dokumentnamen
    base = "incremental_headings" if cfg.respect_headings else "incremental"
    if cfg.respect_headings and cfg.heading_mode != _DEFAULT_HEADING_MODE:
        base += f"_{cfg.heading_mode}"
    if not cfg.smart_split:
        base += "_midpoint"
    if not cfg.filter_low_info:
        # Eigener Schluessel, sonst ueberschreibt die Filter-Ablation den
        # Hauptcache — und genau ihr Vergleich trennt Filter- von Grenzeffekt.
        base += "_nofilter"
    return base


def cache_key(cfg: ChunkerConfig, **overrides) -> str:
    """Name der Cache-Datei fuer diesen Arm, ohne Dokumentnamen und Endung."""
    return _suffix(_with(cfg, **overrides))


def label(cfg: ChunkerConfig, **overrides) -> str:
    """Name dieses Arms in der Ergebnistabelle."""
    cfg = _with(cfg, **overrides)
    if cfg.mode == "window":
        return "llm_window"
    return "llm_" + _suffix(cfg)
