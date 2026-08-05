from __future__ import annotations

from ..interfaces import LLMClient
from .prompt import IncrementalBoundaryPrompt


class IncrementalBoundaryDetector:
    """
    Reads the text sequentially: accumulates sentences into the current
    chunk and asks the LLM for each candidate group whether it continues
    the same topic or starts a new one. Boundaries are drawn by the LLM
    itself, at sentence granularity — no pre-grouped mini-chunks.
    """

    def __init__(
        self,
        client: LLMClient,
        prompt: IncrementalBoundaryPrompt | None = None,
        step_sentences: int = 3,
        max_chunk_sentences: int = 20,
        max_chunk_chars: int | None = None,
        verbose: bool = False,
    ) -> None:
        self.client = client
        self.prompt = prompt or IncrementalBoundaryPrompt()
        self.step_sentences = step_sentences
        self.max_chunk_sentences = max_chunk_sentences
        self.max_chunk_chars = max_chunk_chars
        self.verbose = verbose

    def _over_limit(self, current: list[str]) -> bool:
        """Close the chunk when it hits the sentence cap or the char cap."""
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

            # Safety limit: close the chunk without asking the LLM
            if self._over_limit(current):
                chunks.append(" ".join(current))
                current = list(candidate)
                continue

            if self._same_topic(current, candidate):
                current.extend(candidate)
            else:
                chunks.append(" ".join(current))
                current = list(candidate)

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
