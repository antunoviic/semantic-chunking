from __future__ import annotations

import dataclasses
from pathlib import Path

from llm_semantic_chunker import ChunkerConfig, LLMChunker, OllamaClient
from llm_semantic_chunker.vectorstore import VectorStore

from eval.question_generator import QuestionGenerator
from eval.result_reporter import ResultReporter
from eval.strategy_evaluator import StrategyEvaluator, build_parent_child

from ..chunk_cache import ChunkCache
from ..document_reader import DocumentReader
from .strategies import ChunkSets, ParentChild, StrategyCollector
from .config import EvalConfig
from .variant import cache_key


class Pipeline:

    def __init__(self, document: str,
                 chunker: ChunkerConfig,
                 evaluation: EvalConfig = EvalConfig()) -> None:
        self._file_path = document
        self._chunker_config = chunker
        self._eval = evaluation

        self._client = OllamaClient()
        self._cache = ChunkCache()
        self._reader = DocumentReader(document)
        self._reporter = ResultReporter()
        self._strategies = StrategyCollector(
            cache=self._cache, config=chunker, file_path=document,
            client=self._client, rechunk=evaluation.rechunk)

    def run(self) -> None:
        doc_stem = Path(self._file_path).stem

        print(f"Reading: {self._file_path}")
        text = self._reader.extract_text()
        print(f"Extracted {len(text)} chars from {self._reader.page_count} page(s)\n")

        llm_chunks = self._llm_chunks(text)

        #chunking after that, questions and retireval
        if self._eval.chunk_only:
            lens = [len(c) for c in llm_chunks]
            print(f"\n[chunk-only] {len(llm_chunks)} chunks cached "
                  f"(Ø {sum(lens)//max(1,len(lens))}, max {max(lens)}) — evaluation skipped.")
            return

        qa_pairs = QuestionGenerator().load(
            doc_stem, self._eval.max_questions, questions_file=self._eval.questions_file)
        self._report_anchor_loss(llm_chunks, qa_pairs)

        print("\n[eval] Initializing vector store (Ollama embeddings)...", flush=True)
        store = VectorStore(persist_dir="./chroma_db_eval")
        print("[eval] Vector store ready.", flush=True)

        chunk_sets, parent_child = self._strategies.collect(text, llm_chunks, store)
        results = self._evaluate(chunk_sets, parent_child, qa_pairs, store)
        self._write_reports(results, doc_stem)

    def _llm_chunks(self, text: str) -> list[str]:
        cfg = self._chunker_config
        key = cache_key(cfg)
        if not self._eval.rechunk:
            cached = self._cache.load(self._file_path, enriched=False, variant=key)
            if cached:
                return cached

        print(f"[chunker] Running LLM chunking (mode={cfg.mode})...")
        chunker = LLMChunker(client=self._client, config=cfg)
        chunks = chunker.chunk(text)
        if not chunks:
            raise RuntimeError("Chunking produced 0 chunks — cannot continue.")

        # config parameters
        params = dataclasses.asdict(cfg)
        params["boundary_stats"] = dict(chunker.boundary_stats)
        self._cache.save(self._file_path, chunks, enriched=False,
                         variant=key, params=params)
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
        evaluator = StrategyEvaluator(store, k=self._eval.top_k)
        print(f"\n[eval] Running retrieval tests (k={self._eval.top_k}) "
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

            #parent-child on baselines
            for base in sorted(chunk_sets):
                if not base.startswith(("recursive_matched", "fixed_matched")):
                    continue
                base_children, base_parents = build_parent_child(chunk_sets[base])
                if not base_children:
                    continue
                print(f"  Evaluating: {base}_parentchild...")
                results.append(evaluator.evaluate(
                    f"{base}_parentchild", base_children, qa_pairs,
                    display_texts=base_parents))
        return results

    def _write_reports(self, results: list[dict], doc_stem: str) -> None:
        self._reporter.print_table(results, self._eval.top_k)
        self._reporter.save_json(results, doc_stem)
        self._reporter.save_markdown(results, self._eval.top_k, doc_stem)
        self._reporter.save_charts(results, self._eval.top_k, doc_stem)
