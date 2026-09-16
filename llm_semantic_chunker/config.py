from __future__ import annotations

import warnings
from dataclasses import dataclass, fields
from typing import Optional

#borders
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

    mode: str = "incremental"

    # incremental
    step_sentences: int = 3
    #sentences per border decision
    max_chunk_sentences: int = 20
    #maximum
    max_chunk_chars: Optional[int] = None
    respect_headings: bool = True
    #border on each heading
    heading_mode: str = "regex"
    """ "regex" (sentence-based), "lines" (line-based),
    "hybrid" (line-based and llm-call)."""
    smart_split: bool = True

    # --- window ---
    window_size: int = 10
    #what llm sees
    step_size: int = 5
    #steps per window
    sentences_per_mini_chunk: int = 3

    # shared for bothn modes
    filter_low_info: bool = True
    enrich: bool = False
    language: Optional[str] = None
    #language for chunks, automatic
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
