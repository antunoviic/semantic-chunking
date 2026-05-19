from __future__ import annotations

import gc
from pathlib import Path

from llm_chunker import LLMChunker, QwenClient
from llm_chunker.vectorstore import VectorStore

from eval.question_generator import QuestionGenerator
from eval.result_reporter import ResultReporter
from eval.strategy_evaluator import StrategyEvaluator, build_strategies, build_semantic_lc

from .chunk_cache import ChunkCache
from .pdf_reader import PDFReader


class ChunkingPipeline:
    """
    Orchestrates the full pipeline in one run:
      1. Extract text from PDF
      2. LLM-chunk (or load from cache)
      3. Generate evaluation questions
      4. Evaluate all strategies
      5. Store LLM chunks in ChromaDB
      6. Print + save results
      7. (Optional) interactive query loop
    """

    def __init__(
        self,
        pdf_path: str,
        rechunk: bool = False,
        enrich: bool = False,
        top_k: int = 3,
        max_questions: int = 50,
        regen_questions: bool = False,
    ) -> None:
        self._pdf_path      = pdf_path
        self._rechunk       = rechunk
        self._enrich        = enrich
        self._top_k         = top_k
        self._max_questions = max_questions
        self._regen_q       = regen_questions

        self._client   = QwenClient()
        self._cache    = ChunkCache()
        self._reader   = PDFReader(pdf_path)
        self._reporter = ResultReporter()

    def run(self, query_mode: bool = False) -> None:
        pdf_stem = Path(self._pdf_path).stem

        # 1. Extract text
        print(f"Reading: {self._pdf_path}")
        text = self._reader.extract_text()
        print(f"Extracted {len(text)} chars from {self._reader.page_count} pages\n")

        # 2. LLM chunks
        llm_chunks = self._get_llm_chunks(text)

        # 3. Questions
        generator = QuestionGenerator(self._client)
        qa_pairs  = generator.generate(llm_chunks, pdf_stem, self._max_questions, self._regen_q)

        # 4. Build all strategy chunks
        print("\n[chunking] Building baseline strategies...")
        all_chunks = build_strategies(text)

        # semantic_lc loads a HuggingFace model — build and release it before
        # VectorStore loads its own model instance to avoid a memory conflict
        print("  Building semantic_lc (loads embedding model)...")
        all_chunks["semantic_lc"] = build_semantic_lc(text)

        all_chunks["llm"] = llm_chunks

        enriched = self._cache.load(self._pdf_path, enriched=True)
        if enriched:
            all_chunks["llm_enriched"] = enriched
        else:
            print("  [llm_enriched] No enriched cache — run with --enrich to generate.")

        for name, chunks in all_chunks.items():
            print(f"  {name:<15}: {len(chunks)} chunks")

        # Force memory cleanup before loading the embedding model in VectorStore
        gc.collect()

        # 5. Evaluate (VectorStore loads its own embedding model here)
        print("\n[eval] Initializing vector store (loading embedding model)...", flush=True)
        store     = VectorStore(persist_dir="./chroma_db_eval")
        print("[eval] Vector store ready.", flush=True)
        evaluator = StrategyEvaluator(store, k=self._top_k)
        print(f"\n[eval] Running retrieval tests (k={self._top_k}) over {len(qa_pairs)} questions...\n")

        results = []
        for name, chunks in all_chunks.items():
            print(f"  Evaluating: {name}...")
            results.append(evaluator.evaluate(name, chunks, qa_pairs))

        # 6. Store LLM chunks in main ChromaDB
        print("\nStoring LLM chunks in ChromaDB...")
        main_store = VectorStore(persist_dir="./chroma_db")
        main_store.add_chunks("llm_semantic", llm_chunks, source=Path(self._pdf_path).name)

        # 7. Report
        self._reporter.print_table(results, self._top_k)
        self._reporter.save_json(results, pdf_stem)
        self._reporter.save_charts(results, self._top_k, pdf_stem)

        # 8. Optional query loop
        if query_mode:
            self._query_loop(main_store)

    def _get_llm_chunks(self, text: str) -> list[str]:
        if not self._rechunk:
            cached = self._cache.load(self._pdf_path, enriched=self._enrich)
            if cached is not None and len(cached) > 0:
                return cached

        print("[chunker] Running LLM chunking...")
        chunker = LLMChunker(
            client=self._client,
            window_size=10,
            sentences_per_mini_chunk=3,
            filter_low_info=True,
            enrich=self._enrich,
            verbose=True,
        )
        chunks = chunker.chunk(text)
        if not chunks:
            print("\n[ERROR] LLM chunking produced 0 chunks.")
            print("        Possible causes:")
            print("        - All chunks were removed by the low-info filter")
            print("        - PDF text extraction failed (scanned/image PDF?)")
            print("        Try: python -m app.main <pdf> --rechunk (to retry)")
            self._cache.save(self._pdf_path, chunks, enriched=self._enrich)
            raise RuntimeError("Chunking produced 0 chunks — cannot continue.")
        self._cache.save(self._pdf_path, chunks, enriched=self._enrich)
        return chunks

    def _query_loop(self, store: VectorStore) -> None:
        print(f"\n{'='*60}")
        print("Ask questions about the document. Type 'quit' to exit.\n")
        while True:
            query = input("Question: ").strip()
            if not query or query.lower() in ("quit", "exit", "q"):
                break
            results = store.query("llm_semantic", query, k=3)
            print()
            for i, r in enumerate(results, 1):
                preview = r.chunk_text[:200].replace("\n", " ")
                print(f"  [{i}] distance={r.distance:.3f}")
                print(f"      {preview}...")
            print()
