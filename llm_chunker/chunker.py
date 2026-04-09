from __future__ import annotations

import re
from typing import Optional

from .llm_client import QwenClient
from .prompts import BoundaryPrompt, LowInfoPrompt


def _paragraph_split(text: str) -> list[str]:
    """Split text on paragraph boundaries (double newlines)."""
    normalized = re.sub(r'\n{2,}', '\n\n', text)
    parts = re.split(r'\n\n', normalized)
    return [p.strip() for p in parts if p.strip()]


def _sentence_split(text: str) -> list[str]:
    """Split a paragraph into sentences. Handles English conventions."""
    protected = text

    # Protect decimal numbers (1.5, 33.7, 0.72)
    protected = re.sub(r'(\d)\.(\d)', r'\1<DOT>\2', protected)

    # Protect common English abbreviations
    protected = re.sub(
        r'\b(Mr|Mrs|Ms|Dr|Prof|Sr|Jr|Inc|Corp|Ltd|Co|vs|etc|approx|est|Fig|No|Vol|Rev|Gen|Gov)\.',
        r'\1<DOT>', protected, flags=re.IGNORECASE
    )

    # Protect initials and Latin abbreviations (U.S., e.g., i.e.)
    protected = re.sub(r'\b([A-Za-z])\.([A-Za-z])\.', r'\1<DOT>\2<DOT>', protected)
    protected = re.sub(r'\b(e\.g|i\.e|a\.m|p\.m)\.',
                       r'\1<DOT>', protected, flags=re.IGNORECASE)

    # Split on sentence-ending punctuation followed by space + uppercase
    parts = re.split(r'(?<=[.!?])\s+(?=[A-Z])', protected)

    # Restore dots
    return [p.replace('<DOT>', '.').strip() for p in parts if p.strip()]


def _make_mini_chunks(text: str, sentences_per_chunk: int = 3) -> list[str]:
    """
    Split on paragraphs first, then group long paragraphs
    into sentence groups of N. Short paragraphs stay whole.
    """
    paragraphs = _paragraph_split(text)
    mini_chunks = []

    for para in paragraphs:
        sentences = _sentence_split(para)
        if len(sentences) <= sentences_per_chunk:
            mini_chunks.append(para)
        else:
            for i in range(0, len(sentences), sentences_per_chunk):
                group = sentences[i:i + sentences_per_chunk]
                mini_chunks.append(" ".join(group))

    return mini_chunks


class LLMChunker:
    """
    Splits text into semantically coherent chunks using an LLM
    with a sliding context window approach.

    Pipeline:
      1. Pre-split text into mini-chunks (sentence groups)
      2. Slide a window over mini-chunks, ask LLM for boundaries
      3. Assemble final chunks from mini-chunks between boundaries
      4. (Optional) Filter out low-information chunks
    """

    def __init__(
        self,
        client: Optional[QwenClient] = None,
        boundary_prompt: Optional[BoundaryPrompt] = None,
        low_info_prompt: Optional[LowInfoPrompt] = None,
        filter_low_info: bool = True,
        window_size: int = 10,
        step_size: int = 5,
        sentences_per_mini_chunk: int = 3,
        verbose: bool = False,
    ) -> None:
        self.client = client or QwenClient()
        self.boundary_prompt = boundary_prompt or BoundaryPrompt()
        self.low_info_prompt = low_info_prompt or LowInfoPrompt()
        self.filter_low_info = filter_low_info
        self.window_size = window_size
        self.step_size = step_size
        self.sentences_per_mini_chunk = sentences_per_mini_chunk
        self.verbose = verbose

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(f"[chunker] {msg}")

    def chunk(self, text: str) -> list[str]:
        """Run the full chunking pipeline."""
        mini_chunks = _make_mini_chunks(text, self.sentences_per_mini_chunk)
        self._log(f"Pre-split into {len(mini_chunks)} mini-chunks")

        if self.verbose:
            for i, mc in enumerate(mini_chunks):
                self._log(f"  mini[{i}]: {mc[:80]}...")

        if len(mini_chunks) <= 2:
            return [text.strip()] if text.strip() else []

        boundaries = self._find_boundaries(mini_chunks)
        self._log(f"Boundaries found at indices: {boundaries}")

        chunks = self._assemble_chunks(mini_chunks, boundaries)
        self._log(f"Assembled {len(chunks)} chunks")

        if self.filter_low_info:
            before = len(chunks)
            chunks = self._remove_low_info(chunks)
            self._log(f"Filter removed {before - len(chunks)} low-info chunks")

        return chunks

    def _find_boundaries(self, mini_chunks: list[str]) -> list[int]:
        """Slide a window over mini-chunks, ask LLM where topics change."""
        all_boundaries: set[int] = set()
        pos = 0

        while pos < len(mini_chunks):
            window = mini_chunks[pos:pos + self.window_size]
            if len(window) <= 1:
                break

            tagged = self._tag_window(window, offset=pos)
            messages = self.boundary_prompt.as_messages(tagged)
            raw = self.client.chat(messages)
            self._log(f"Window [{pos}:{pos+len(window)}] LLM response: {raw!r}")

            local_bounds = self._parse_boundaries(raw, offset=pos,
                                                   max_idx=pos + len(window) - 1)
            all_boundaries.update(local_bounds)

            if local_bounds:
                pos = max(local_bounds)
            else:
                pos += self.step_size

        return sorted(all_boundaries)

    def _tag_window(self, window: list[str], offset: int) -> str:
        """Wrap each mini-chunk in numbered XML tags."""
        parts = []
        for i, mc in enumerate(window):
            idx = offset + i
            parts.append(f"<chunk_{idx}>{mc}</chunk_{idx}>")
        return "\n\n".join(parts)

    def _parse_boundaries(self, raw: str, offset: int, max_idx: int) -> list[int]:
        """Extract valid boundary indices from the LLM response."""
        numbers = re.findall(r'\d+', raw)
        bounds = []
        for n in numbers:
            idx = int(n)
            if offset < idx <= max_idx:
                bounds.append(idx)
        return bounds

    def _assemble_chunks(self, mini_chunks: list[str], boundaries: list[int]) -> list[str]:
        """Join mini-chunks between boundary points into final chunks."""
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

    def _remove_low_info(self, chunks: list[str]) -> list[str]:
        """Ask the LLM whether each chunk contains useful information."""
        result = []
        for chunk in chunks:
            messages = self.low_info_prompt.as_messages(chunk)
            response = self.client.chat(messages).strip().upper()
            if response.startswith("YES"):
                result.append(chunk)
            else:
                print(f"[filter] removed: {chunk[:60]}...")
        return result