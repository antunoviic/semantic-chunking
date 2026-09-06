from __future__ import annotations

from pathlib import Path
from typing import Optional

from llm_chunker import LLMChunker, QwenClient
from llm_chunker.vectorstore import VectorStore

from eval.question_generator import QuestionGenerator
from eval.result_reporter import ResultReporter
from eval.strategy_evaluator import StrategyEvaluator

from ..chunk_cache import ChunkCache
from ..document_reader import DocumentReader
from .strategies import ChunkSets, ParentChild, StrategyCollector
from .variant import ChunkVariant


class Pipeline:

    def __init__(
        self,
        file_path: str,
        rechunk: bool = False,
        enrich: bool = False,
        incremental: bool = True,
        step_sentences: int = 3,
        max_chunk_sentences: int = 20,
        max_chunk_chars: Optional[int] = None,
        respect_headings: bool = True,
        heading_mode: str = "regex",
        smart_split: bool = True,
        filter_low_info: bool = True,
        chunk_only: bool = False,
        top_k: int = 3,
        max_questions: int = 1000,
        questions_file: Optional[str] = None,
    ) -> None:
        self._file_path = file_path
        self._rechunk = rechunk
        self._step_sents = step_sentences
        self._max_chunk_sents = max_chunk_sentences
        self._max_chunk_chars = max_chunk_chars
        self._filter_low_info = filter_low_info
        self._chunk_only = chunk_only
        self._top_k = top_k
        self._max_questions = max_questions
        self._questions_file = questions_file

        self._variant = ChunkVariant(incremental=incremental,
                                     respect_headings=respect_headings,
                                     smart_split=smart_split,
                                     heading_mode=heading_mode)
        self._client = QwenClient()
        self._cache = ChunkCache()
        self._reader = DocumentReader(file_path)
        self._reporter = ResultReporter()
        self._strategies = StrategyCollector(
            cache=self._cache, variant=self._variant, file_path=file_path,
            client=self._client, rechunk=rechunk, enrich=enrich)

    def run(self) -> None:
        doc_stem = Path(self._file_path).stem

        print(f"Reading: {self._file_path}")
        text = self._reader.extract_text()
        print(f"Extracted {len(text)} chars from {self._reader.page_count} page(s)\n")

        llm_chunks = self._llm_chunks(text)

        #chunking after that, questions and retireval
        if self._chunk_only:
            lens = [len(c) for c in llm_chunks]
            print(f"\n[chunk-only] {len(llm_chunks)} chunks cached "
                  f"(Ø {sum(lens)//max(1,len(lens))}, max {max(lens)}) — evaluation skipped.")
            return

        qa_pairs = QuestionGenerator().load(
            doc_stem, self._max_questions, questions_file=self._questions_file)
        self._report_anchor_loss(llm_chunks, qa_pairs)

        print("\n[eval] Initializing vector store (Ollama embeddings)...", flush=True)
        store = VectorStore(persist_dir="./chroma_db_eval")
        print("[eval] Vector store ready.", flush=True)

        chunk_sets, parent_child = self._strategies.collect(text, llm_chunks, store)
        results = self._evaluate(chunk_sets, parent_child, qa_pairs, store)
        self._write_reports(results, doc_stem)

    def _llm_chunks(self, text: str) -> list[str]:
        key = self._variant.key()
        if not self._rechunk:
            cached = self._cache.load(self._file_path, enriched=False, variant=key)
            if cached:
                return cached
    #llm chunker in his variants
        mode = "incremental" if self._variant.incremental else "window"
        print(f"[chunker] Running LLM chunking (mode={mode})...")
        chunker = LLMChunker(
            client=self._client,
            mode=mode,
            window_size=10,
            sentences_per_mini_chunk=3,
            step_sentences=self._step_sents,
            max_chunk_sentences=self._max_chunk_sents,
            max_chunk_chars=self._max_chunk_chars,
            respect_headings=self._variant.respect_headings,
            heading_mode=self._variant.heading_mode,
            smart_split=self._variant.smart_split,
            filter_low_info=self._filter_low_info,
            enrich=False,
            verbose=True,
        )
        chunks = chunker.chunk(text)
        if not chunks:
            raise RuntimeError("Chunking produced 0 chunks — cannot continue.")
        self._cache.save(self._file_path, chunks, enriched=False, variant=key)
        return chunks

    @staticmethod
    def _report_anchor_loss(chunks: list[str], qa_pairs: list[dict]) -> int:
        #reports on low filter removal
        blob = " ".join(" ".join(c.split()) for c in chunks)
        lost = [q for q in qa_pairs if " ".join(q["source_text"].split()) not in blob]
        n = len(qa_pairs) or 1
        print(f"[anchors] {len(qa_pairs) - len(lost)}/{len(qa_pairs)} Anker in den LLM-Chunks "
              f"vorhanden — {len(lost)} verloren ({len(lost) * 100 // n} %)")
        for q in lost[:3]:
            print(f"          fehlt: {q['question'][:70]}")
        return len(lost)

    def _evaluate(self, chunk_sets: ChunkSets, parent_child: ParentChild,
                  qa_pairs: list[dict], store: VectorStore) -> list[dict]:
        evaluator = StrategyEvaluator(store, k=self._top_k)
        print(f"\n[eval] Running retrieval tests (k={self._top_k}) "
              f"over {len(qa_pairs)} questions...\n")

        results = []
        for name, chunks in chunk_sets.items():
            print(f"  Evaluating: {name}...")
            results.append(evaluator.evaluate(name, chunks, qa_pairs))

        children, parents = parent_child
        if children:
            print("  Evaluating: llm_incremental_parentchild...")
            results.append(evaluator.evaluate(
                "llm_incremental_parentchild", children, qa_pairs, display_texts=parents))
        return results

    def _write_reports(self, results: list[dict], doc_stem: str) -> None:
        self._reporter.print_table(results, self._top_k)
        self._reporter.save_json(results, doc_stem)
        self._reporter.save_markdown(results, self._top_k, doc_stem)
        self._reporter.save_charts(results, self._top_k, doc_stem)
