from __future__ import annotations

from .detector import IncrementalBoundaryDetector
from .heading_detection import (HeadingOnlyPrompt, heading_sentence_indices,
                                looks_like_heading_candidate)

#how much of text the llm gets to see full heading without text
_PROBE_CHARS = 120


class HybridHeadingBoundaryDetector(IncrementalBoundaryDetector):

    def __init__(self, *args, heading_prompt: HeadingOnlyPrompt | None = None,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.heading_prompt = heading_prompt or HeadingOnlyPrompt()
        self._heading_indices: set[int] = set()

    def reset_stats(self) -> None:
        super().reset_stats()
        self.boundary_stats["heading_regex"] = 0
        self.boundary_stats["heading_llm"] = 0
        # how often llm call made
        self.boundary_stats["heading_llm_checks"] = 0

    def detect_and_assemble(self, sentences: list[str], raw_text: str | None = None) -> list[str]:
        if not raw_text:
            raise ValueError(
                "HybridHeadingBoundaryDetector braucht raw_text (Rohtext vor der "
                "Satztokenisierung) — die Regex-Baseline arbeitet zeilenbasiert."
            )
        self._heading_indices = heading_sentence_indices(raw_text, sentences)
        if self.verbose:
            print(f"[hybrid-heading] Regex-Baseline: {len(self._heading_indices)} "
                  f"Satz-Indizes als Ueberschrift erkannt")
        return super().detect_and_assemble(sentences)

    def _same_topic(self, current: list[str], candidate: list[str],
                    start_idx: int | None = None) -> bool:
        # 1. Regex-Treffer -> harte Grenze, kostenlos
        if start_idx is not None and start_idx in self._heading_indices:
            if self.verbose:
                print(f"[hybrid-heading] Regex-Grenze bei Satz {start_idx}: "
                      f"{candidate[0][:80]!r}")
            self.boundary_stats["heading_regex"] += 1
            return False

        probe = " ".join(candidate)[:_PROBE_CHARS].strip()
        if probe and looks_like_heading_candidate(probe):
            self.boundary_stats["heading_llm_checks"] += 1
            raw = self.client.chat(self.heading_prompt.as_messages(probe)).strip()
            if self.heading_prompt.parse(raw):
                self.boundary_stats["heading_llm"] += 1
                if self.verbose:
                    print(f"[hybrid-heading] LLM-Grenze bei Satz {start_idx}: {probe[:80]!r}")
                return False

        return super()._same_topic(current, candidate, start_idx)
