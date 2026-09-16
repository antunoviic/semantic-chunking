from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class EvalConfig:
    questions_file: Optional[str] = None
    max_questions: int = 1000
    top_k: int = 3
    chunk_only: bool = False
    rechunk: bool = False
