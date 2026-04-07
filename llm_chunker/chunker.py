from __future__ import annotations

import re
from typing import Optional

from .llm_client import QwenClient
from .prompts import BoundaryPrompt, LowInfoPrompt


def _paragraph_split(text: str) -> list[str]:
    """
    Split text into natural segments using paragraph boundaries.
    PDF text typically has meaningful line/paragraph breaks.
    """
    # Normalize: replace multiple newlines with a marker
    # Split on: double newlines, section markers (· · ·), page dividers (— N —)
    normalized = re.sub(r'\n{2,}', '\n\n', text)

    # Split on paragraph breaks, section markers, and page markers
    parts = re.split(
        r'\n\n'                         # double newline (paragraph break)
        r'|(?=· · · .+? · · ·)'        # section markers like · · · ABSCHNITT 1 · · ·
        r'|(?=— \d+ —)',               # page markers like — 1 —
        normalized
    )
    return [p.strip() for p in parts if p.strip()]


def _sentence_split(text: str) -> list[str]:
    """Split a paragraph into sentences. Handles German conventions."""
    # Protect common non-sentence dots (1.000, Abb., Nr., etc.)
    protected = text
    protected = re.sub(r'(\d)\.(\d)', r'\1<DOT>\2', protected)
    protected = re.sub(r'(Abb|Nr|Dr|Prof|bzw|etc|ca|vgl|z\.B|d\.h|u\.a|Dipl|Ing)\.',
                       r'\1<DOT>', protected, flags=re.IGNORECASE)

    # Split on sentence-ending punctuation
    parts = re.split(r'(?<=[.!?])\s+(?=[A-ZÄÖÜA-Z])', protected)

    # Restore dots
    return [p.replace('<DOT>', '.').strip() for p in parts if p.strip()]


def _make_mini_chunks(text: str, sentences_per_chunk: int = 3) -> list[str]:
    """
    Create mini-chunks by first splitting on paragraphs,
    then splitting long paragraphs into sentence groups.
    Short paragraphs become their own mini-chunk.
    """
    paragraphs = _paragraph_split(text)
    mini_chunks = []

    for para in paragraphs:
        sentences = _sentence_split(para)
        if len(sentences) <= sentences_per_chunk:
            # Short paragraph → keep as one mini-chunk
            mini_chunks.append(para)
        else:
            # Long paragraph → group sentences
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
      2. Slide a window over mini-chunks, asking the LLM for boundaries
      3. Merge mini-chunks between boundaries into final chunks
      4. (Optional) Filter out low-information chunks
    """

    def __init__(
        self,
        client: Optional[QwenClient] = None,
        boundary_prompt: Optional[BoundaryPrompt] = None,
        low_info_prompt: Optional[LowInfoPrompt] = None,
        filter_low_info: bool = True,
        window_size: int = 10,        # mini-chunks per LLM call
        step_size: int = 5,           # how far to advance if no boundary found
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
        """Return a list of semantic chunks for the given text."""
        mini_chunks = _make_mini_chunks(text, self.sentences_per_mini_chunk)
        self._log(f"Pre-split into {len(mini_chunks)} mini-chunks")

        if self.verbose:
            for i, mc in enumerate(mini_chunks):
                self._log(f"  mini[{i}]: {mc[:80]}...")

        if len(mini_chunks) <= 2:
            return [text.strip()] if text.strip() else []

        boundaries = self._find_boundaries(mini_chunks)
        self._log(f"Boundaries found at indices: {boundaries}")

        chunks = self._merge_at_boundaries(mini_chunks, boundaries)
        self._log(f"Merged into {len(chunks)} chunks")

        if self.filter_low_info:
            before = len(chunks)
            chunks = self._remove_low_info(chunks)
            self._log(f"Filter removed {before - len(chunks)} low-info chunks")

        return chunks

    def _find_boundaries(self, mini_chunks: list[str]) -> list[int]:
        """
        Slide a window over tagged mini-chunks and ask the LLM
        where semantic boundaries are. Returns sorted boundary indices.
        """
        all_boundaries: set[int] = set()
        pos = 0

        while pos < len(mini_chunks):
            window = mini_chunks[pos:pos + self.window_size]
            if len(window) <= 1:
                break

            # Build tagged text for this window
            tagged = self._tag_window(window, offset=pos)
            messages = self.boundary_prompt.as_messages(tagged)
            raw = self.client.chat(messages)
            self._log(f"Window [{pos}:{pos+len(window)}] LLM response: {raw!r}")
            local_bounds = self._parse_boundaries(raw, offset=pos,
                                                   max_idx=pos + len(window) - 1)

            all_boundaries.update(local_bounds)

            # Advance to last boundary in this window, or step forward
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
        """Extract boundary indices from the LLM response."""
        # Find all numbers in the response
        numbers = re.findall(r'\d+', raw)
        bounds = []
        for n in numbers:
            idx = int(n)
            # Only accept valid indices within the current window
            if offset < idx <= max_idx:
                bounds.append(idx)
        return bounds

    def _merge_at_boundaries(self, mini_chunks: list[str], boundaries: list[int]) -> list[str]:
        """Merge mini-chunks between boundary points into final chunks."""
        if not boundaries:
            return [" ".join(mini_chunks)]

        chunks = []
        prev = 0
        for b in boundaries:
            segment = mini_chunks[prev:b]
            if segment:
                chunks.append(" ".join(segment))
            prev = b
        # Last segment
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