from __future__ import annotations

import re
import unicodedata
from pathlib import Path

from llm_chunker import LLMChunker, QwenClient
from llm_chunker.vectorstore import VectorStore

from eval.question_generator import QuestionGenerator
from eval.rag_evaluator import RAGEvaluator
from eval.result_reporter import ResultReporter
from eval.strategy_evaluator import StrategyEvaluator, build_strategies, build_semantic_lc

from .chunk_cache import ChunkCache
from .pdf_reader import PDFReader


class ChunkingPipeline:

    def __init__(
        self,
        pdf_path: str,
        rechunk: bool = False,
        enrich: bool = False,
        top_k: int = 3,
        max_questions: int = 50,
        regen_questions: bool = False,
        rag_eval: bool = False,
    ) -> None:
        self._pdf_path      = pdf_path
        self._rechunk       = rechunk
        self._enrich        = enrich
        self._top_k         = top_k
        self._max_questions = max_questions
        self._regen_q       = regen_questions
        self._rag_eval      = rag_eval

        self._client   = QwenClient()
        self._cache    = ChunkCache()
        self._reader   = PDFReader(pdf_path)
        self._reporter = ResultReporter()

    @staticmethod
    def _safe_name(stem: str) -> str:
        normalized = unicodedata.normalize("NFKD", stem).encode("ascii", "ignore").decode()
        sanitized  = re.sub(r"[^a-zA-Z0-9._-]", "_", normalized).strip("._-")
        return sanitized or "collection"

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

        # 4. Vector store (Ollama embeddings — one model for everything)
        print("\n[eval] Initializing vector store (Ollama embeddings)...", flush=True)
        store = VectorStore(persist_dir="./chroma_db_eval")
        print("[eval] Vector store ready.", flush=True)

        # 5. Build all strategy chunks
        print("\n[chunking] Building baseline strategies...")
        all_chunks = build_strategies(text)

        print("  Building semantic_lc (Ollama nomic-embed-text)...")
        all_chunks["semantic_lc"] = build_semantic_lc(text, store._embedding_fn)

        all_chunks["llm"] = llm_chunks

        enriched = self._cache.load(self._pdf_path, enriched=True)
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
        collection_map: dict[str, str] = {}
        for name, chunks in all_chunks.items():
            print(f"  Evaluating: {name}...")
            results.append(evaluator.evaluate(name, chunks, qa_pairs))
            collection_map[name] = f"eval_{name}"

        # 7. Store LLM chunks in main ChromaDB
        print("\nStoring LLM chunks in ChromaDB...")
        main_store = VectorStore(persist_dir="./chroma_db")
        main_store.add_chunks(self._safe_name(pdf_stem), llm_chunks, source=Path(self._pdf_path).name)

        # 8. Retrieval report
        self._reporter.print_table(results, self._top_k)
        self._reporter.save_json(results, pdf_stem)
        self._reporter.save_charts(results, self._top_k, pdf_stem)

        # 9. End-to-end RAG evaluation (optional)
        if self._rag_eval:
            self._run_rag_eval(store, collection_map, qa_pairs, pdf_stem)

        # 10. Optional query loop
        if query_mode:
            self._query_loop(main_store, pdf_stem)

    def _run_rag_eval(
        self,
        store: VectorStore,
        collection_map: dict[str, str],
        qa_pairs: list[dict],
        pdf_stem: str,
    ) -> None:
        print("\n[rag-eval] Starting end-to-end RAG evaluation...")
        print("[rag-eval] Generator: Qwen (local) — upload eval_results/*.md to Claude for scoring")

        rag_evaluator = RAGEvaluator(
            generator=self._client,
            evaluator=self._client,
        )
        rag_results = rag_evaluator.evaluate_all(
            store=store,
            strategy_collections=collection_map,
            qa_pairs=qa_pairs,
            k=self._top_k,
            pdf_stem=pdf_stem,
        )
        RAGEvaluator.print_table(rag_results)

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
            self._cache.save(self._pdf_path, chunks, enriched=self._enrich)
            raise RuntimeError("Chunking produced 0 chunks — cannot continue.")
        self._cache.save(self._pdf_path, chunks, enriched=self._enrich)
        return chunks

    def _query_loop(self, store: VectorStore, collection: str) -> None:
        print(f"\n{'='*60}")
        print("Ask questions about the document. Type 'quit' to exit.\n")
        while True:
            query = input("Question: ").strip()
            if not query or query.lower() in ("quit", "exit", "q"):
                break
            results = store.query(collection, query, k=self._top_k)
            print()
            for i, r in enumerate(results, 1):
                preview = r.chunk_text[:200].replace("\n", " ")
                print(f"  [{i}] distance={r.distance:.3f}")
                print(f"      {preview}...")
            print()
