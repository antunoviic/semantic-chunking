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
            try:
                data = json.loads(qa_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                data = []
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
            # Pick a specific sentence as source so any chunk size can contain it
            source_sentence = self._pick_source_sentence(chunk_text)

            messages = [
                {
                    "role": "system",
                    "content": (
                        "Generate exactly one specific factual question whose answer "
                        "is a concrete fact from the given text: a name, number, quote, "
                        "or specific claim. The question must be unanswerable without "
                        "this exact passage — avoid questions about general topics or "
                        "themes. Do not ask 'what does the speaker think about X' or "
                        "'what is the main idea'. Instead ask for specific facts: "
                        "'According to Naval, how long does X take?', "
                        "'What term does the speaker use for Y?', "
                        "'What specific example is given for Z?'. "
                        "Output only the question, nothing else."
                    ),
                },
                {"role": "user", "content": source_sentence},
            ]
            question = self._client.chat(messages).strip()
            question = re.sub(r"^(Question:|Q:)\s*", "", question, flags=re.IGNORECASE).strip()
            qa_pairs.append({
                "question":    question,
                "source_text": source_sentence,
                "chunk_index": chunk_idx,
            })
            print(f"  [{idx+1}/{len(sampled)}] {question[:80]}")

        qa_path.write_text(json.dumps(qa_pairs, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[questions] Saved to {qa_path}")
        return qa_pairs

    @staticmethod
    def _pick_source_sentence(chunk_text: str, min_len: int = 60) -> str:
        """Return the most informative sentence from a chunk.

        Picks the longest sentence from the middle third of the chunk to avoid
        titles and trailing context sentences that are too generic.
        """
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', chunk_text) if len(s.strip()) >= min_len]
        if not sentences:
            return chunk_text[:400]
        # focus on middle third to avoid chapter headings at start
        lo = len(sentences) // 3
        hi = max(lo + 1, 2 * len(sentences) // 3)
        candidates = sentences[lo:hi] or sentences
        return max(candidates, key=len)
