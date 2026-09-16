from __future__ import annotations

from llm_semantic_chunker.post_processors import ChunkEnricher
from llm_semantic_chunker.prompts import EnrichmentPrompt
from llm_semantic_chunker.vectorstore import VectorStore

from eval.strategy_evaluator import (build_parent_child, build_semantic_lc,
                                     build_strategies)

from llm_semantic_chunker import ChunkerConfig

from ..chunk_cache import ChunkCache
from .variant import cache_key, label

ChunkSets = dict[str, list[str]]
ParentChild = tuple[list[str], list[str]]


class StrategyCollector:

    def __init__(self, cache: ChunkCache, config: ChunkerConfig, file_path: str,
                 client, rechunk: bool = False, enrich: bool = False) -> None:
        self._cache = cache
        self._config = config
        self._file_path = file_path
        self._client = client
        self._rechunk = rechunk
        self._enrich = enrich

    def collect(self, text: str, llm_chunks: list[str],
                store: VectorStore) -> tuple[ChunkSets, ParentChild]:
        print("\n[chunking] Building baseline strategies...")
        match_len = round(sum(map(len, llm_chunks)) / max(1, len(llm_chunks)))
        chunk_sets = build_strategies(text, match_len=match_len)

        print("  Building semantic_lc (Ollama embeddings)...")
        chunk_sets["semantic_lc"] = build_semantic_lc(text, store._embedding_fn)
        chunk_sets[label(self._config)] = llm_chunks

        chunk_sets.update(self._other_mode())
        chunk_sets.update(self._ablation_counterparts())
        chunk_sets.update(self._enriched_row(llm_chunks))

        parent_child: ParentChild = ([], [])
        if self._config.mode == "incremental":
            parent_child = build_parent_child(llm_chunks)
            print(f"  parent_child   : {len(parent_child[0])} children "
                  f"aus {len(llm_chunks)} parents")

        for name, chunks in chunk_sets.items():
            print(f"  {name:<15}: {len(chunks)} chunks")
        return chunk_sets, parent_child

    # window or incremental
    def _other_mode(self) -> ChunkSets:
        """Das jeweils andere Verfahren, sofern gecacht — window neben
        incremental und umgekehrt."""
        other = "window" if self._config.mode == "incremental" else "incremental"
        name = label(self._config, mode=other)
        chunks = self._cache.load(self._file_path, variant=cache_key(self._config, mode=other))
        return {name: chunks} if chunks else {}

    def _ablation_counterparts(self) -> ChunkSets:
        if self._config.mode == "window":
            return {}
        cfg = self._config
        pairs = [
            # Heading-Ablation, with or without headings
            (label(cfg, respect_headings=not cfg.respect_headings),
             cache_key(cfg, respect_headings=not cfg.respect_headings)),
            # Split-Ablation: LLM chooses split or just the middle
            (label(cfg, smart_split=not cfg.smart_split),
             cache_key(cfg, smart_split=not cfg.smart_split)),
            (label(cfg, respect_headings=False, filter_low_info=False),
             cache_key(cfg, respect_headings=False, filter_low_info=False)),
        ]

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

    def _enriched_row(self, llm_chunks: list[str]) -> ChunkSets:
        #[Topic:]-Variant 
        variant = cache_key(self._config)
        if self._enrich:
            return {"llm_enriched": self._build_enriched(llm_chunks, variant)}
        cached = self._cache.load(self._file_path, enriched=True, variant=variant)
        if cached:
            return {"llm_enriched": cached}
        print("  [llm_enriched] No enriched cache — run with --enrich to generate.")
        return {}

    def _build_enriched(self, plain_chunks: list[str], variant: str) -> list[str]:
        #labels for every strategy
        if not self._rechunk:
            cached = self._cache.load(self._file_path, enriched=True, variant=variant)
            if cached:
                return cached
        print("[chunker] Enriching chunks with [Topic: ...] prefix...")
        enriched = ChunkEnricher(self._client, EnrichmentPrompt(), verbose=True).process(plain_chunks)
        self._cache.save(self._file_path, enriched, enriched=True, variant=variant)
        return enriched
