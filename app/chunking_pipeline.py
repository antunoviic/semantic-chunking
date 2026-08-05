from __future__ import annotations

from pathlib import Path

from llm_chunker import LLMChunker, QwenClient
from llm_chunker.post_processors import ChunkEnricher
from llm_chunker.prompts import EnrichmentPrompt
from llm_chunker.vectorstore import VectorStore

from eval.question_generator import QuestionGenerator
from eval.result_reporter import ResultReporter
from eval.strategy_evaluator import StrategyEvaluator, build_strategies, build_semantic_lc

from .chunk_cache import ChunkCache
from .document_reader import DocumentReader


class ChunkingPipeline:

    def __init__(
        self,
        file_path: str,
        rechunk: bool = False,
        enrich: bool = False,
        incremental: bool = True,
        step_sentences: int = 3,
        max_chunk_sentences: int = 20,
        max_chunk_chars: Optional[int] = None,
        top_k: int = 3,
        max_questions: int = 50,
        questions_file: Optional[str] = None,
    ) -> None:
        self._file_path     = file_path
        self._rechunk       = rechunk
        self._enrich        = enrich
        self._incremental   = incremental
        self._step_sents    = step_sentences
        self._max_chunk_sents = max_chunk_sentences
        self._max_chunk_chars = max_chunk_chars
        self._top_k         = top_k
        self._max_questions = max_questions
        self._questions_file = questions_file

        self._client   = QwenClient()
        self._cache    = ChunkCache()
        self._reader   = DocumentReader(file_path)
        self._reporter = ResultReporter()

    def run(self) -> None:
        doc_stem = Path(self._file_path).stem

        # 1. Extract text
        print(f"Reading: {self._file_path}")
        text = self._reader.extract_text()
        print(f"Extracted {len(text)} chars from {self._reader.page_count} page(s)\n")

        # 2. LLM chunks
        llm_chunks = self._get_llm_chunks(text)

        # 3. Questions
        generator = QuestionGenerator()
        qa_pairs  = generator.load(doc_stem, self._max_questions, questions_file=self._questions_file)

        # 4. Vector store (Ollama embeddings — one model for everything)
        print("\n[eval] Initializing vector store (Ollama embeddings)...", flush=True)
        store = VectorStore(persist_dir="./chroma_db_eval")
        print("[eval] Vector store ready.", flush=True)

        # 5. Build all strategy chunks
        print("\n[chunking] Building baseline strategies...")
        all_chunks = build_strategies(text)

        print("  Building semantic_lc (Ollama embeddings)...")
        all_chunks["semantic_lc"] = build_semantic_lc(text, store._embedding_fn)

        llm_label = "llm_incremental" if self._incremental else "llm_window"
        all_chunks[llm_label] = llm_chunks

        # If the other mode was chunked before, include it from cache for comparison
        other_label   = "llm_window" if self._incremental else "llm_incremental"
        other_variant = "" if self._incremental else "incremental"
        other_chunks  = self._cache.load(self._file_path, variant=other_variant)
        if other_chunks:
            all_chunks[other_label] = other_chunks

        if self._enrich:
            all_chunks["llm_enriched"] = self._get_enriched_chunks(llm_chunks)
        else:
            enriched = self._cache.load(self._file_path, enriched=True,
                                        variant="incremental" if self._incremental else "")
            if enriched:
                all_chunks["llm_enriched"] = enriched
            else:
                print("  [llm_enriched] No enriched cache — run with --enrich to generate.")

        for name, chunks in all_chunks.items():
            print(f"  {name:<15}: {len(chunks)} chunks")

        # 6. Retrieval evaluation (Hit@K, MRR, Avg Dist)
        evaluator = StrategyEvaluator(store, k=self._top_k)
        print(f"\n[eval] Running retrieval tests (k={self._top_k}) over {len(qa_pairs)} questions...\n")

        results = []
        for name, chunks in all_chunks.items():
            print(f"  Evaluating: {name}...")
            results.append(evaluator.evaluate(name, chunks, qa_pairs))

        # 7. Retrieval report
        self._reporter.print_table(results, self._top_k)
        self._reporter.save_json(results, doc_stem)
        self._reporter.save_markdown(results, self._top_k, doc_stem)
        self._reporter.save_charts(results, self._top_k, doc_stem)

    def _get_llm_chunks(self, text: str) -> list[str]:
        """Produce the plain (un-enriched) LLM chunks for the base window/incremental row."""
        variant = "incremental" if self._incremental else ""
        if not self._rechunk:
            cached = self._cache.load(self._file_path, enriched=False, variant=variant)
            if cached is not None and len(cached) > 0:
                return cached

        mode = "incremental" if self._incremental else "window"
        print(f"[chunker] Running LLM chunking (mode={mode})...")
        chunker = LLMChunker(
            client=self._client,
            mode=mode,
            window_size=10,
            sentences_per_mini_chunk=3,
            step_sentences=self._step_sents,
            max_chunk_sentences=self._max_chunk_sents,
            max_chunk_chars=self._max_chunk_chars,
            filter_low_info=True,
            enrich=False,
            verbose=True,
        )
        chunks = chunker.chunk(text)
        if not chunks:
            raise RuntimeError("Chunking produced 0 chunks — cannot continue.")
        self._cache.save(self._file_path, chunks, enriched=False, variant=variant)
        return chunks

    def _get_enriched_chunks(self, plain_chunks: list[str]) -> list[str]:
        """Enrich the SAME plain chunks with a [Topic: ...] prefix.

        Applied on top of the plain chunks so both variants share identical
        boundaries — the only difference is the prefix, which isolates the
        effect of enrichment on retrieval.
        """
        variant = "incremental" if self._incremental else ""
        if not self._rechunk:
            cached = self._cache.load(self._file_path, enriched=True, variant=variant)
            if cached is not None and len(cached) > 0:
                return cached

        print("[chunker] Enriching chunks with [Topic: ...] prefix...")
        enricher = ChunkEnricher(self._client, EnrichmentPrompt(), verbose=True)
        enriched = enricher.process(plain_chunks)
        self._cache.save(self._file_path, enriched, enriched=True, variant=variant)
        return enriched
