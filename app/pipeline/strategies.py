from __future__ import annotations

from typing import TYPE_CHECKING

from llm_semantic_chunker import ChunkerConfig

if TYPE_CHECKING:                      # annotation only
    from eval.vectorstore import VectorStore

from eval.strategy_evaluator import (build_parent_child, build_semantic_lc,
                                     build_strategies)

from ..chunk_cache import ChunkCache
from .variant import cache_key, label

ChunkSets = dict[str, list[str]]
ParentChild = tuple[list[str], list[str]]


class StrategyCollector:
    """Assembles every chunk set that goes into one evaluation report.

    The LLM arm of the current run is passed in. The rule-based baselines are
    built from the text; every other LLM arm comes from the chunk cache, and only
    if it was produced by the code that is running now (ChunkCache.load skips
    stale caches). Each cached counterpart differs from the current
    configuration in exactly one setting, see `_ablation_counterparts`.
    """

    def __init__(self, cache: ChunkCache, config: ChunkerConfig, file_path: str) -> None:
        self._cache = cache
        self._config = config
        self._file_path = file_path

    def collect(self, text: str, llm_chunks: list[str],
                store: "VectorStore") -> tuple[ChunkSets, ParentChild]:
        print("\n[chunking] Building baseline strategies...")
        match_len = round(sum(map(len, llm_chunks)) / max(1, len(llm_chunks)))
        chunk_sets = build_strategies(text, match_len=match_len)

        print("  Building semantic_lc (Ollama embeddings)...")
        chunk_sets["semantic_lc"] = build_semantic_lc(
            text, store._embedding_fn, match_len=match_len,
            max_chars=self._config.max_chunk_chars)
        chunk_sets[label(self._config)] = llm_chunks

        chunk_sets.update(self._other_mode())
        chunk_sets.update(self._ablation_counterparts())

        parent_child: ParentChild = ([], [])
        if self._config.mode == "incremental":
            parent_child = build_parent_child(llm_chunks)
            print(f"  parent_child   : {len(parent_child[0])} children "
                  f"from {len(llm_chunks)} parents")

        for name, chunks in chunk_sets.items():
            print(f"  {name:<15}: {len(chunks)} chunks")
        return chunk_sets, parent_child

    def _other_mode(self) -> ChunkSets:
        """The other mode, if it is cached — window alongside incremental
        and vice versa."""
        other = "window" if self._config.mode == "incremental" else "incremental"
        name = label(self._config, mode=other)
        chunks = self._cache.load(self._file_path, variant=cache_key(self._config, mode=other))
        return {name: chunks} if chunks else {}

    def _ablation_counterparts(self) -> ChunkSets:
        """Every cached arm that differs from the current one in exactly one setting.

        Toggled one at a time: headings on/off, smart vs. midpoint split, low-info
        filter on/off, topic enrichment on/off. Plus the other heading-detection
        modes, each with headings on. Arms whose cache is missing or stale are
        simply absent from the report.
        """
        if self._config.mode == "window":
            return {}
        cfg = self._config
        toggles = [
            dict(respect_headings=not cfg.respect_headings),
            dict(smart_split=not cfg.smart_split),
            dict(filter_low_info=not cfg.filter_low_info),
            dict(enrich=not cfg.enrich),
        ]
        pairs = [(label(cfg, **t), cache_key(cfg, **t)) for t in toggles]
        for hmode in ("regex", "lines", "hybrid"):
            if cfg.respect_headings and hmode == cfg.heading_mode:
                continue
            pairs.append((label(cfg, respect_headings=True, heading_mode=hmode),
                          cache_key(cfg, respect_headings=True, heading_mode=hmode)))

        out: ChunkSets = {}
        for name, key in pairs:
            chunks = self._cache.load(self._file_path, variant=key)
            if chunks:
                out[name] = chunks
        return out
