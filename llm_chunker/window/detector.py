from __future__ import annotations

import re

from ..interfaces import BasePrompt, LLMClient


class BoundaryDetector:
    """
    Slides a window over mini-chunks, asks an LLM where topic boundaries
    occur, and assembles the final chunks.
    """

    def __init__(
        self,
        client: LLMClient,
        prompt: BasePrompt,
        window_size: int = 10,
        step_size: int = 5,
        verbose: bool = False,
    ) -> None:
        self.client = client
        self.prompt = prompt
        self.window_size = window_size
        self.step_size = step_size
        self.verbose = verbose

    def detect_and_assemble(self, mini_chunks: list[str]) -> list[str]:
        """Detect boundaries and assemble final chunks from mini-chunks."""
        if len(mini_chunks) <= 2:
            return [" ".join(mini_chunks)] if mini_chunks else []

        boundaries = self._find_boundaries(mini_chunks)
        if self.verbose:
            print(f"[boundary_detector] Boundaries at: {boundaries}")

        return self._assemble_chunks(mini_chunks, boundaries)

    def _find_boundaries(self, mini_chunks: list[str]) -> list[int]:
        all_boundaries: set[int] = set()
        pos = 0
        while pos < len(mini_chunks):
            window = mini_chunks[pos:pos + self.window_size]
            if len(window) <= 1:
                break
            tagged = self._tag_window(window, offset=pos)
            messages = self.prompt.as_messages(tagged)
            raw = self.client.chat(messages)
            if self.verbose:
                print(f"[boundary_detector] Window [{pos}:{pos+len(window)}] -> {raw!r}")
            local_bounds = self._parse_boundaries(raw, offset=pos, max_idx=pos + len(window) - 1)
            all_boundaries.update(local_bounds)
            if local_bounds:
                pos = max(local_bounds)
            else:
                pos += self.step_size
        return sorted(all_boundaries)

    @staticmethod
    def _tag_window(window: list[str], offset: int) -> str:
        parts = [f"<chunk_{offset + i}>{mc}</chunk_{offset + i}>" for i, mc in enumerate(window)]
        return "\n\n".join(parts)

    @staticmethod
    def _parse_boundaries(raw: str, offset: int, max_idx: int) -> list[int]:
        # Only treat as NONE if the entire response is "NONE" (not if NONE appears in an explanation)
        if re.fullmatch(r'\s*NONE\s*', raw, re.IGNORECASE):
            return []
        # Extract standalone numbers not embedded in words
        numbers = re.findall(r'(?<!\w)(\d+)(?!\w)', raw)
        return [int(n) for n in numbers if offset < int(n) <= max_idx]

    @staticmethod
    def _assemble_chunks(mini_chunks: list[str], boundaries: list[int]) -> list[str]:
        if not boundaries:
            return [" ".join(mini_chunks)]
        chunks = []
        prev = 0
        for b in boundaries:
            segment = mini_chunks[prev:b]
            if segment:
                chunks.append(" ".join(segment))
            prev = b
        remaining = mini_chunks[prev:]
        if remaining:
            chunks.append(" ".join(remaining))
        return [c.strip() for c in chunks if c.strip()]
