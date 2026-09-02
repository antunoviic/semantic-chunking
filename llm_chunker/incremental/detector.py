from __future__ import annotations

import re

from ..interfaces import LLMClient
from .prompt import IncrementalBoundaryPrompt, SplitPointPrompt


class IncrementalBoundaryDetector:

    def __init__(
        self,
        client: LLMClient,
        prompt: IncrementalBoundaryPrompt | None = None,
        split_prompt: SplitPointPrompt | None = None,
        step_sentences: int = 3,
        max_chunk_sentences: int = 20,
        max_chunk_chars: int | None = None,
        smart_split: bool = True,
        verbose: bool = False,
    ) -> None:
        self.client = client
        self.prompt = prompt or IncrementalBoundaryPrompt()
        self.split_prompt = split_prompt or SplitPointPrompt()
        self.step_sentences = step_sentences
        self.max_chunk_sentences = max_chunk_sentences
        self.max_chunk_chars = max_chunk_chars
        # Ablation: False = am Limit einfach in der Mitte trennen, ohne das LLM
        # nach der besten Stelle zu fragen. Trennt den Effekt des Groessendeckels
        # vom Effekt der LLM-Trennstellenwahl.
        self.smart_split = smart_split
        self.verbose = verbose
        self.boundary_stats: dict[str, int] = {}
        self.reset_stats()

    def _over_limit(self, current: list[str]) -> bool:
        if len(current) >= self.max_chunk_sentences:
            return True
        if self.max_chunk_chars and len(" ".join(current)) >= self.max_chunk_chars:
            return True
        return False

    def reset_stats(self) -> None:
        """Zaehlt, WOHER die Chunk-Grenzen stammen.

        Zentrale Kennzahl fuer die Arbeit: Kommt der Grossteil der Grenzen vom
        Groessenlimit statt von der semantischen Entscheidung, ist das Verfahren
        im Kern ein Fixed-Size-Chunker mit LLM-Overhead.
        """
        self.boundary_stats = {"semantic": 0, "size_cap": 0, "heading": 0, "end": 0}

    def detect_and_assemble(self, sentences: list[str]) -> list[str]:
        self.reset_stats()
        return self._run(sentences)

    def _run(self, sentences: list[str]) -> list[str]:
        """Eigentliche Schleife. Getrennt von detect_and_assemble, damit die
        Heading-Variante sie je Abschnitt aufrufen kann, ohne die Zaehler zu
        ueberschreiben."""
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
                self.boundary_stats["semantic"] += 1
                current = list(candidate)
                continue

            # Size cap reached: split at the best boundary (not a hard cut),
            # keep the remainder for the next chunk. Loop in case still oversized.
            while self._over_limit(current):
                idx = self._best_split(current)
                chunks.append(" ".join(current[:idx]))
                self.boundary_stats["size_cap"] += 1
                current = current[idx:]

        if current:
            chunks.append(" ".join(current))
            self.boundary_stats["end"] += 1
        return [c.strip() for c in chunks if c.strip()]

    def _same_topic(self, current: list[str], candidate: list[str]) -> bool:
        messages = self.prompt.as_messages(" ".join(current), " ".join(candidate))
        raw = self.client.chat(messages).strip().upper()
        if self.verbose:
            print(f"[incremental] chunk={len(current)} sents, candidate={len(candidate)} sents -> {raw!r}")
        # Only an explicit NO starts a new chunk; unclear answers keep merging
        return not raw.startswith("NO")

    def _best_split(self, current: list[str]) -> int:
        #Ask the LLM for the most sensible split point in an oversized chunk.

        n = len(current)
        if n <= 1:
            return 1
        if not self.smart_split:
            idx = max(1, n // 2)          # Ablation: naiver Mittel-Split, kein LLM
            if self.verbose:
                print(f"[incremental] size cap hit ({n} sents) -> midpoint split at {idx}")
            return idx
        raw = self.client.chat(self.split_prompt.as_messages(current)).strip()
        if self.verbose:
            print(f"[incremental] size cap hit ({n} sents) -> best split at {raw!r}")
        m = re.search(r"\d+", raw)
        if m:
            idx = int(m.group()) - 1          # 1-based sentence -> 0-based split index
            if 1 <= idx <= n - 1:
                return idx
        return max(1, n // 2)                  # fallback: balanced split in the middle
