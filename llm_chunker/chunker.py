from __future__ import annotations

from typing import Optional

from .incremental import IncrementalBoundaryDetector, IncrementalBoundaryPrompt
from .interfaces import ChunkPostProcessor, LLMClient
from .llm_client import QwenClient
from .post_processors import ChunkEnricher, LowInfoFilter
from .prompts import EnrichmentPrompt, LowInfoPrompt
from .text_splitter import TextSplitter
from .window import BoundaryDetector, BoundaryPrompt


class LLMChunker:
    """
    Orchestrates the full semantic chunking pipeline:
      1. TextSplitter      — split text into mini-chunks (window mode) or sentences (incremental mode)
      2. Boundary detection — mode="window": sliding window over pre-grouped mini-chunks
                              mode="incremental": LLM reads sequentially and draws its own boundaries
      3. post_processors   — optional filter and enrichment steps (OCP: extend without modifying)
    """

    def __init__(
        self,
        client: Optional[LLMClient] = None,
        boundary_prompt: Optional[BoundaryPrompt] = None,
        low_info_prompt: Optional[LowInfoPrompt] = None,
        enrichment_prompt: Optional[EnrichmentPrompt] = None,
        incremental_prompt: Optional[IncrementalBoundaryPrompt] = None,
        filter_low_info: bool = True,
        enrich: bool = False,
        mode: str = "window",
        window_size: int = 10,
        step_size: int = 5,
        sentences_per_mini_chunk: int = 3,
        step_sentences: int = 3,
        max_chunk_sentences: int = 20,
        max_chunk_chars: Optional[int] = None,
        language: Optional[str] = None,
        verbose: bool = False,
    ) -> None:
        if mode not in ("window", "incremental"):
            raise ValueError(f"Unknown mode: {mode!r}. Use 'window' or 'incremental'.")
        self._client: LLMClient = client or QwenClient()
        self._mode = mode
        self._max_chunk_chars = max_chunk_chars
        self._verbose = verbose

        self._splitter = TextSplitter(sentences_per_chunk=sentences_per_mini_chunk, language=language)

        if mode == "incremental":
            self._detector = IncrementalBoundaryDetector(
                client=self._client,
                prompt=incremental_prompt or IncrementalBoundaryPrompt(),
                step_sentences=step_sentences,
                max_chunk_sentences=max_chunk_sentences,
                max_chunk_chars=max_chunk_chars,
                verbose=verbose,
            )
        else:
            self._detector = BoundaryDetector(
                client=self._client,
                prompt=boundary_prompt or BoundaryPrompt(),
                window_size=window_size,
                step_size=step_size,
                verbose=verbose,
            )

        self._post_processors: list[ChunkPostProcessor] = []
        if filter_low_info:
            self._post_processors.append(
                LowInfoFilter(
                    client=self._client,
                    prompt=low_info_prompt or LowInfoPrompt(),
                    verbose=verbose,
                )
            )
        if enrich:
            self._post_processors.append(
                ChunkEnricher(
                    client=self._client,
                    prompt=enrichment_prompt or EnrichmentPrompt(),
                    verbose=verbose,
                )
            )

    def chunk(self, text: str) -> list[str]:
        """Run the full chunking pipeline."""
        if self._mode == "incremental":
            units = self._splitter.split_sentences(text)
            if self._verbose:
                print(f"[chunker] Split into {len(units)} sentences (incremental mode)")
        else:
            units = self._splitter.make_mini_chunks(text)
            if self._verbose:
                print(f"[chunker] Pre-split into {len(units)} mini-chunks")

        chunks = self._detector.detect_and_assemble(units)
        if self._verbose:
            print(f"[chunker] Assembled {len(chunks)} chunks")

        if self._max_chunk_chars:
            before = len(chunks)
            chunks = self._cap_sizes(chunks)
            if self._verbose:
                print(f"[chunker] Size cap ({self._max_chunk_chars} chars): {before} -> {len(chunks)} chunks")

        for processor in self._post_processors:
            before = len(chunks)
            chunks = processor.process(chunks)
            if self._verbose:
                print(f"[chunker] {processor.__class__.__name__}: {before} -> {len(chunks)} chunks")

        return chunks

    def _cap_sizes(self, chunks: list[str]) -> list[str]:
        """Split any chunk longer than max_chunk_chars at sentence boundaries.

        Applies to both modes; for window mode this is the only size safeguard
        (the sliding-window detector has no size limit of its own). A single
        sentence longer than the cap is kept whole rather than split mid-sentence.
        """
        cap = self._max_chunk_chars
        capped: list[str] = []
        for chunk in chunks:
            if len(chunk) <= cap:
                capped.append(chunk)
                continue
            current = ""
            for sentence in self._splitter.split_sentences(chunk):
                if current and len(current) + 1 + len(sentence) > cap:
                    capped.append(current)
                    current = sentence
                else:
                    current = f"{current} {sentence}".strip()
            if current:
                capped.append(current)
        return capped
