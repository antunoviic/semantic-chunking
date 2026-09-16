from __future__ import annotations

from typing import Optional

from .incremental import (
    HeadingAwareBoundaryDetector,
    HybridHeadingBoundaryDetector,
    IncrementalBoundaryDetector,
    IncrementalBoundaryPrompt,
)
from .config import ChunkerConfig
from .interfaces import ChunkPostProcessor, LLMClient
from .llm_client import OllamaClient
from .post_processors import ChunkEnricher, LowInfoFilter
from .prompts import EnrichmentPrompt, LowInfoPrompt
from .text_splitter import TextSplitter
from .window import BoundaryDetector, BoundaryPrompt
import logging

from ._logging import enable_console_logging, get_logger

logger = get_logger(__name__)


class LLMChunker:
    #Splits text into semantically coherent chunks using a local LLM.

        #LLMChunker(client=OllamaClient(), max_chunk_chars=1200)
        #LLMChunker(client=OllamaClient(), config=ChunkerConfig(mode="window"))

    def __init__(
        self,
        client: Optional[LLMClient] = None,
        config: Optional[ChunkerConfig] = None,
        *,
        boundary_prompt: Optional[BoundaryPrompt] = None,
        low_info_prompt: Optional[LowInfoPrompt] = None,
        enrichment_prompt: Optional[EnrichmentPrompt] = None,
        incremental_prompt: Optional[IncrementalBoundaryPrompt] = None,
        **settings,
    ) -> None:
        if config is not None and settings:
            raise TypeError(
                "pass either config=ChunkerConfig(...) or the individual "
                f"settings, not both (got {', '.join(sorted(settings))})")
        self._config = config if config is not None else ChunkerConfig(**settings)
        self._config.warn_about_ignored()

        if self._config.verbose:
            # the only thing `verbose` does: a DEBUG console handler on the
            # package logger; every module logs unconditionally through `logging`
            enable_console_logging(logging.DEBUG)

        self._client: LLMClient = client or OllamaClient()
        self._owns_client = client is None
        self.boundary_stats: dict[str, int] = {}

        self._splitter = TextSplitter(
            sentences_per_chunk=self._config.sentences_per_mini_chunk,
            language=self._config.language)
        self._detector = self._build_detector(boundary_prompt, incremental_prompt)
        self._post_processors = self._build_post_processors(
            low_info_prompt, enrichment_prompt)

    @property
    def config(self) -> ChunkerConfig:
        #settings this chunker was built with
        return self._config

    def _build_detector(self, boundary_prompt, incremental_prompt):
        #Picks the detector the mode and heading_mode call for

        cfg = self._config
        if cfg.mode == "window":
            return BoundaryDetector(
                client=self._client,
                prompt=boundary_prompt or BoundaryPrompt(),
                window_size=cfg.window_size,
                step_size=cfg.step_size,
            )

        extra: dict = {}
        if not cfg.respect_headings:
            detector_cls = IncrementalBoundaryDetector
        elif cfg.heading_mode in ("hybrid", "lines"):
            detector_cls = HybridHeadingBoundaryDetector
            extra["use_llm_fallback"] = cfg.heading_mode == "hybrid"
        else:
            detector_cls = HeadingAwareBoundaryDetector
        return detector_cls(
            client=self._client,
            prompt=incremental_prompt or IncrementalBoundaryPrompt(),
            step_sentences=cfg.step_sentences,
            max_chunk_sentences=cfg.max_chunk_sentences,
            max_chunk_chars=cfg.max_chunk_chars,
            smart_split=cfg.smart_split,
            **extra,
        )

    def _build_post_processors(self, low_info_prompt, enrichment_prompt
                               ) -> list[ChunkPostProcessor]:
        cfg = self._config
        out: list[ChunkPostProcessor] = []
        if cfg.filter_low_info:
            out.append(LowInfoFilter(client=self._client,
                                     prompt=low_info_prompt or LowInfoPrompt()))
        if cfg.enrich:
            out.append(ChunkEnricher(client=self._client,
                                     prompt=enrichment_prompt or EnrichmentPrompt()))
        return out

    def chunk(self, text: str) -> list[str]:
        #chunking itself
        if self._config.mode == "incremental":
            units = self._splitter.split_sentences(text)
            logger.debug(f"[chunker] Split into {len(units)} sentences (incremental mode)")
        else:
            units = self._splitter.make_mini_chunks(text)
            logger.debug(f"[chunker] Pre-split into {len(units)} mini-chunks")

        # raw_text for hybrid in incremental
        raw_text = text if self._config.mode == "incremental" else None
        chunks = self._detector.detect_and_assemble(units, raw_text=raw_text)
        self.boundary_stats = dict(getattr(self._detector, "boundary_stats", {}))
        logger.info("[chunker] Assembled %d chunks", len(chunks))
        if self.boundary_stats:
            total = sum(self.boundary_stats.values()) or 1
            parts = "  ".join(f"{k}={v} ({v * 100 // total} %)"
                              for k, v in self.boundary_stats.items() if v)
            logger.info("[chunker] boundary sources: %s", parts)

        if self._config.max_chunk_chars and self._config.mode != "incremental":
            before = len(chunks)
            chunks = self._cap_sizes(chunks)
            logger.debug(f"[chunker] Size cap ({self._config.max_chunk_chars} chars): {before} -> {len(chunks)} chunks")

        for processor in self._post_processors:
            before = len(chunks)
            chunks = processor.process(chunks)
            logger.debug(f"[chunker] {processor.__class__.__name__}: {before} -> {len(chunks)} chunks")

        return chunks

    def close(self) -> None:
        """Release the HTTP client, if this chunker created one itself."""
        if self._owns_client:
            closer = getattr(self._client, "close", None)
            if callable(closer):
                closer()

    def __enter__(self) -> "LLMChunker":
        return self

    def __exit__(self, *_) -> None:
        self.close()

    def _cap_sizes(self, chunks: list[str]) -> list[str]:
        #Split any chunk longer than max_chunk_chars at sentence boundaries

        cap = self._config.max_chunk_chars
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
