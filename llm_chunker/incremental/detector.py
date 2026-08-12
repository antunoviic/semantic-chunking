from __future__ import annotations

import re

from ..interfaces import LLMClient
from .prompt import IncrementalBoundaryPrompt, SplitPointPrompt


class IncrementalBoundaryDetector:
    """
    Reads the text sequentially: accumulates sentences into the current
    chunk and asks the LLM for each candidate group whether it continues
    the same topic or starts a new one. Boundaries are drawn by the LLM
    itself, at sentence granularity — no pre-grouped mini-chunks.

    When a chunk reaches the size cap, it is NOT hard-cut at the end: the LLM
    is asked for the most sensible split point within the accumulated sentences
    (SplitPointPrompt). Only the part before that boundary becomes a chunk; the
    remainder is carried forward. This prevents mid-topic cuts on similar content.
    """

    def __init__(
        self,
        client: LLMClient,
        prompt: IncrementalBoundaryPrompt | None = None,
        split_prompt: SplitPointPrompt | None = None,
        step_sentences: int = 3,
        max_chunk_sentences: int = 20,
        max_chunk_chars: int | None = None,
        verbose: bool = False,
    ) -> None:
        self.client = client
        self.prompt = prompt or IncrementalBoundaryPrompt()
        self.split_prompt = split_prompt or SplitPointPrompt()
        self.step_sentences = step_sentences
        self.max_chunk_sentences = max_chunk_sentences
        self.max_chunk_chars = max_chunk_chars
        self.verbose = verbose

    def _over_limit(self, current: list[str]) -> bool:
        """True once the chunk hits the sentence cap or the char cap."""
        if len(current) >= self.max_chunk_sentences:
            return True
        if self.max_chunk_chars and len(" ".join(current)) >= self.max_chunk_chars:
            return True
        return False

    def detect_and_assemble(self, sentences: list[str]) -> list[str]:
        """Input: individual sentences (not pre-grouped). Output: semantic chunks."""
        chunks: list[str] = []
        current: list[str] = []
        pos = 0

        while pos < len(sentences):
            candidate = sentences[pos:pos + self.step_sentences]
            pos += self.step_sentences

            if not current:
                current = list(candidate)
                continue

            if self._same_topic(current, candidate):
                current.extend(candidate)
            else:
                # Natural topic boundary -> close the chunk here.
                chunks.append(" ".join(current))
                current = list(candidate)
                continue

            # Size cap reached: split at the best boundary (not a hard cut),
            # keep the remainder for the next chunk. Loop in case still oversized.
            while self._over_limit(current):
                idx = self._best_split(current)
                chunks.append(" ".join(current[:idx]))
                current = current[idx:]

        if current:
            chunks.append(" ".join(current))
        return [c.strip() for c in chunks if c.strip()]

    def _same_topic(self, current: list[str], candidate: list[str]) -> bool:
        messages = self.prompt.as_messages(" ".join(current), " ".join(candidate))
        raw = self.client.chat(messages).strip().upper()
        if self.verbose:
            print(f"[incremental] chunk={len(current)} sents, candidate={len(candidate)} sents -> {raw!r}")
        # Only an explicit NO starts a new chunk; unclear answers keep merging
        return not raw.startswith("NO")

    def _best_split(self, current: list[str]) -> int:
        """Ask the LLM for the most sensible split point in an oversized chunk.

        Returns a 0-based index in [1, len-1] where the SECOND part begins, so
        current[:idx] is saved as a chunk and current[idx:] is carried forward.
        A single unsplittable sentence returns 1 (it becomes its own chunk).
        """
        n = len(current)
        if n <= 1:
            return 1
        raw = self.client.chat(self.split_prompt.as_messages(current)).strip()
        if self.verbose:
            print(f"[incremental] size cap hit ({n} sents) -> best split at {raw!r}")
        m = re.search(r"\d+", raw)
        if m:
            idx = int(m.group()) - 1          # 1-based sentence -> 0-based split index
            if 1 <= idx <= n - 1:
                return idx
        return max(1, n // 2)                  # fallback: balanced split in the middle
