from __future__ import annotations

import json
from pathlib import Path
from typing import Optional


class QuestionGenerator:
    """Loads a QA evaluation set (question + verbatim source_text) from JSON.

    Question sets are curated externally (a strong LLM writes them, then each
    source_text is verified to be a verbatim substring of the document) because
    auto-generating them with the local model produced shallow, unreliable
    questions. Sets live at eval_cache/<doc_stem>_questions.json, or are passed
    explicitly via questions_file.

    Format: [{"question": "...", "source_text": "..."}]
    """

    def __init__(self, cache_dir: Path = Path("./eval_cache")) -> None:
        self._dir = cache_dir

    def load(
        self,
        doc_stem: str,
        max_questions: int = 50,
        questions_file: Optional[str] = None,
    ) -> list[dict]:
        # An explicit file wins over the doc-stem convention.
        path = Path(questions_file) if questions_file else self._dir / f"{doc_stem}_questions.json"
        if not path.exists():
            raise FileNotFoundError(
                f"No question set found at {path}. Provide one via --questions-file, "
                f"or place it at eval_cache/{doc_stem}_questions.json."
            )
        data = json.loads(path.read_text(encoding="utf-8"))[:max_questions]
        print(f"[questions] Loaded {len(data)} questions from {path.name}")
        return data
