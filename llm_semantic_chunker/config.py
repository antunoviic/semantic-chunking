from __future__ import annotations

import warnings
from dataclasses import dataclass, fields
from typing import Optional

# lower bounds for the integer settings; anything below is rejected
_MINIMUMS = {
    "step_sentences": 1,
    "max_chunk_sentences": 1,
    "sentences_per_mini_chunk": 1,
    "window_size": 2,
    "step_size": 1,
}

# which parameters on which mode
_WINDOW_ONLY = ("window_size", "step_size", "sentences_per_mini_chunk")
_INCREMENTAL_ONLY = ("step_sentences", "max_chunk_sentences", "max_chunk_chars",
                     "respect_headings", "heading_mode", "smart_split")


@dataclass(frozen=True)
class ChunkerConfig:
    """Every setting that shapes a chunking run, validated on construction.

    Frozen, so a configuration cannot drift while a run is in progress, and
    self-validating, so a bad value fails before the first model call rather
    than hours into a document.

    The fields fall into three groups: those that only apply to incremental
    mode, those that only apply to window mode, and those shared by both. A
    setting from the wrong group is ignored rather than refused, which
    `warn_about_ignored()` reports — a known weakness of holding both modes in
    one class.
    """

    mode: str = "incremental"

    # --- incremental mode ---
    # sentences per boundary decision
    step_sentences: int = 3
    # hard cap in sentences, regardless of topic continuity
    max_chunk_sentences: int = 20
    # cap in characters, applied at sentence boundaries; None = no cap
    max_chunk_chars: Optional[int] = None
    # force a boundary at detected section headings
    respect_headings: bool = True
    # "regex" (sentence-based), "lines" (line-based), "hybrid" (line-based + LLM check)
    heading_mode: str = "regex"
    # at the size cap, ask the LLM for the best split point instead of cutting in the middle
    smart_split: bool = True

    # --- window mode ---
    # mini-chunks visible to the LLM per boundary decision
    window_size: int = 10
    # how far the window advances per iteration
    step_size: int = 5
    # sentences per mini-chunk
    sentences_per_mini_chunk: int = 3

    # --- shared by both modes ---
    # drop near-empty chunks after assembly (one LLM call per chunk)
    filter_low_info: bool = True
    # prefix each chunk with an LLM-generated [Topic: ...] line (one LLM call per chunk)
    enrich: bool = False
    # sentence-splitter language; auto-detected if None
    language: Optional[str] = None
    # attach a DEBUG console handler to the package logger (see _logging.py)
    verbose: bool = False

    def __post_init__(self) -> None:
        if self.mode not in ("window", "incremental"):
            raise ValueError(f"Unknown mode: {self.mode!r}. Use 'window' or 'incremental'.")
        if self.heading_mode not in ("regex", "lines", "hybrid"):
            raise ValueError(f"Unknown heading_mode: {self.heading_mode!r}. "
                             f"Use 'regex', 'lines' or 'hybrid'.")
        for name, minimum in _MINIMUMS.items():
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
                raise ValueError(f"{name} must be an int >= {minimum}, got {value!r}")
        if self.max_chunk_chars is not None and (not isinstance(self.max_chunk_chars, int)
                                                 or self.max_chunk_chars < 1):
            raise ValueError(f"max_chunk_chars must be a positive int or None, "
                             f"got {self.max_chunk_chars!r}")

    def warn_about_ignored(self) -> None:
        defaults = {f.name: f.default for f in fields(self)}
        candidates = _WINDOW_ONLY if self.mode == "incremental" else _INCREMENTAL_ONLY
        ignored = sorted(n for n in candidates if getattr(self, n) != defaults[n])
        if ignored:
            warnings.warn(f"mode={self.mode!r} ignores these parameters: "
                          f"{', '.join(ignored)}", stacklevel=3)
