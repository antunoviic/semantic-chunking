from __future__ import annotations

from typing import Optional

from .boundary_detector import BoundaryDetector
from .interfaces import ChunkPostProcessor, LLMClient
from .llm_client import QwenClient
from .post_processors import ChunkEnricher, LowInfoFilter
from .prompts import BoundaryPrompt, EnrichmentPrompt, LowInfoPrompt
from .text_splitter import TextSplitter


class LLMChunker:
    """
    Orchestrates the full semantic chunking pipeline:
      1. TextSplitter      — pre-split text into mini-chunks
      2. BoundaryDetector  — detect topic boundaries and assemble chunks
      3. post_processors   — optional filter and enrichment steps (OCP: extend without modifying)
    """

    def __init__(
        self,
        client: Optional[LLMClient] = None,
        boundary_prompt: Optional[BoundaryPrompt] = None,
        low_info_prompt: Optional[LowInfoPrompt] = None,
        enrichment_prompt: Optional[EnrichmentPrompt] = None,
        filter_low_info: bool = True,
        enrich: bool = False,
        window_size: int = 10,
        step_size: int = 5,
        sentences_per_mini_chunk: int = 3,
        language: Optional[str] = None,
        verbose: bool = False,
    ) -> None:
        self._client: LLMClient = client or QwenClient()
        self._verbose = verbose

        self._splitter = TextSplitter(sentences_per_chunk=sentences_per_mini_chunk, language=language)

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
        mini_chunks = self._splitter.make_mini_chunks(text)
        if self._verbose:
            print(f"[chunker] Pre-split into {len(mini_chunks)} mini-chunks")

        chunks = self._detector.detect_and_assemble(mini_chunks)
        if self._verbose:
            print(f"[chunker] Assembled {len(chunks)} chunks")

        for processor in self._post_processors:
            before = len(chunks)
            chunks = processor.process(chunks)
            if self._verbose:
                print(f"[chunker] {processor.__class__.__name__}: {before} -> {len(chunks)} chunks")

        return chunks
