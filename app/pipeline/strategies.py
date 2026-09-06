from __future__ import annotations

from llm_chunker.post_processors import ChunkEnricher
from llm_chunker.prompts import EnrichmentPrompt
from llm_chunker.vectorstore import VectorStore

from eval.strategy_evaluator import (build_parent_child, build_semantic_lc,
                                     build_strategies)

from ..chunk_cache import ChunkCache
from .variant import ChunkVariant

ChunkSets = dict[str, list[str]]
ParentChild = tuple[list[str], list[str]]


class StrategyCollector:

    def __init__(self, cache: ChunkCache, variant: ChunkVariant, file_path: str,
                 client, rechunk: bool = False, enrich: bool = False) -> None:
        self._cache = cache
        self._variant = variant
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
        chunk_sets[self._variant.label()] = llm_chunks

        chunk_sets.update(self._other_mode())
        chunk_sets.update(self._ablation_counterparts())
        chunk_sets.update(self._enriched_row(llm_chunks))

        parent_child: ParentChild = ([], [])
        if self._variant.incremental:
            parent_child = build_parent_child(llm_chunks)
            print(f"  parent_child   : {len(parent_child[0])} children "
                  f"aus {len(llm_chunks)} parents")

        for name, chunks in chunk_sets.items():
            print(f"  {name:<15}: {len(chunks)} chunks")
        return chunk_sets, parent_child

    # window or incremental
    def _other_mode(self) -> ChunkSets:
        incremental = self._variant.incremental
        label = "llm_window" if incremental else "llm_incremental"
        chunks = self._cache.load(self._file_path,
                                  variant="" if incremental else "incremental")
        return {label: chunks} if chunks else {}

    def _ablation_counterparts(self) -> ChunkSets:
        if not self._variant.incremental:
            return {}
        v = self._variant
        other_hmode = "regex" if v.heading_mode != "regex" else "hybrid"
        pairs = [
            # Heading-Ablation, with or without headings
            (v.label(respect_headings=not v.respect_headings),
             v.key(respect_headings=not v.respect_headings)),
            # Split-Ablation: LLM chooses split or just the middle
            (v.label(smart_split=not v.smart_split),
             v.key(smart_split=not v.smart_split)),
        ]

        if v.respect_headings:
            pairs.append((v.label(heading_mode=other_hmode),
                          v.key(heading_mode=other_hmode)))

        out: ChunkSets = {}
        for label, key in pairs:
            chunks = self._cache.load(self._file_path, variant=key)
            if chunks:
                out[label] = chunks
        return out

    def _enriched_row(self, llm_chunks: list[str]) -> ChunkSets:
        #[Topic:]-Variant 
        variant = "incremental" if self._variant.incremental else ""
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
