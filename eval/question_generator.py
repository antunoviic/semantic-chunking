from __future__ import annotations

import json
import re
from pathlib import Path

from llm_chunker.interfaces import LLMClient


class QuestionGenerator:
    """SRP: generates QA pairs from chunks via an LLM, with caching."""

    def __init__(self, client: LLMClient, cache_dir: Path = Path("./eval_cache")) -> None:
        self._client = client
        self._dir = cache_dir

    def generate(
        self,
        chunks: list[str],
        pdf_stem: str,
        max_questions: int = 50,
        regen: bool = False,
    ) -> list[dict]:
        self._dir.mkdir(exist_ok=True)
        qa_path = self._dir / f"{pdf_stem}_questions.json"

        if qa_path.exists() and not regen:
            data = json.loads(qa_path.read_text(encoding="utf-8"))
            if data:
                data = data[:max_questions]
                print(f"[questions] Loaded {len(data)} questions from cache")
                return data
            print("[questions] Cache is empty — regenerating...")

        if qa_path.exists():
            qa_path.unlink()

        step = max(1, len(chunks) // max_questions)
        sampled = [(i, chunks[i]) for i in range(0, len(chunks), step)][:max_questions]
        print(f"[questions] Generating {len(sampled)} questions via Qwen...")

        qa_pairs = []
        for idx, (chunk_idx, chunk_text) in enumerate(sampled):
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You generate exactly one specific question that is answered "
                        "by the given text. Output only the question, nothing else."
                    ),
                },
                {"role": "user", "content": chunk_text},
            ]
            question = self._client.chat(messages).strip()
            question = re.sub(r"^(Question:|Q:)\s*", "", question, flags=re.IGNORECASE).strip()
            qa_pairs.append({
                "question":    question,
                "source_text": chunk_text,
                "chunk_index": chunk_idx,
            })
            print(f"  [{idx+1}/{len(sampled)}] {question[:80]}")

        qa_path.write_text(json.dumps(qa_pairs, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[questions] Saved to {qa_path}")
        return qa_pairs
