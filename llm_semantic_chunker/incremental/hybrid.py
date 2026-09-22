from __future__ import annotations

from .detector import IncrementalBoundaryDetector
from .heading_detection import (HeadingOnlyPrompt, heading_sentence_indices,
                                looks_like_heading_candidate)
from .._logging import get_logger

logger = get_logger(__name__)

# Upper bound on the candidate text handed to the pre-filter. The pre-filter
# itself accepts at most _MAX_HEADING_LEN (90) characters and 12 words, so the
# LLM is only ever asked about a short sentence group; a heading that shares its
# group with a full sentence is not checked at all.
_PROBE_CHARS = 120


class HybridHeadingBoundaryDetector(IncrementalBoundaryDetector):

    def __init__(self, *args, heading_prompt: HeadingOnlyPrompt | None = None,
                 use_llm_fallback: bool = True, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.heading_prompt = heading_prompt or HeadingOnlyPrompt()
        self.use_llm_fallback = use_llm_fallback
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
                "HybridHeadingBoundaryDetector needs raw_text (the text before "
                "sentence tokenisation): the regex baseline works line by line."
            )
        self._heading_indices = heading_sentence_indices(raw_text, sentences)
        # The three logger.debug strings in this file are the last German text in
        # the codebase, and they stay. The cache fingerprint hashes the AST of this
        # module, and a string literal is part of that AST: translating them changes
        # the digest, which invalidates every chunk cache of the reported run.
        # A comment is free, a log message is not.
        logger.debug(f"[hybrid-heading] Regex-Baseline: {len(self._heading_indices)} "
              f"Satz-Indizes als Ueberschrift erkannt")
        return super().detect_and_assemble(sentences)

    def _same_topic(self, current: list[str], candidate: list[str],
                    start_idx: int | None = None) -> bool:
        # regex hit: boundary in front of the group that contains the heading
        if start_idx is not None and self._heading_indices:
            hit = next((i for i in range(start_idx, start_idx + len(candidate))
                        if i in self._heading_indices), None)
            if hit is not None:
                logger.debug(f"[hybrid-heading] Regex-Grenze bei Satz {hit}: "
                      f"{candidate[hit - start_idx][:80]!r}")
                self.boundary_stats["heading_regex"] += 1
                return False

        # LLM check, only for short groups that pass the pre-filter
        if not self.use_llm_fallback:
            return super()._same_topic(current, candidate, start_idx)

        probe = " ".join(candidate)[:_PROBE_CHARS].strip()
        if probe and looks_like_heading_candidate(probe):
            self.boundary_stats["heading_llm_checks"] += 1
            raw = self.client.chat(self.heading_prompt.as_messages(probe)).strip()
            is_heading = self.heading_prompt.parse(raw)
            logger.debug(f"[hybrid-heading] Satz {start_idx} raw={raw[:40]!r} "
                  f"-> {'HEADING' if is_heading else 'PROSE'}: {probe[:60]!r}")
            if is_heading:
                self.boundary_stats["heading_llm"] += 1
                return False

        return super()._same_topic(current, candidate, start_idx)
