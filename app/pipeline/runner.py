from __future__ import annotations

import dataclasses
import hashlib
import platform
from importlib import metadata
from pathlib import Path

import httpx

from typing import TYPE_CHECKING

from llm_semantic_chunker import ChunkerConfig, LLMChunker, OllamaClient

if TYPE_CHECKING:                      # annotations only; no import at runtime
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
            cache=self._cache, config=chunker, file_path=document)

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

        from llm_semantic_chunker.vectorstore import VectorStore

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

        # Everything a reader needs to tell whether two caches are comparable:
        # the settings, what the boundaries came from, the exact input text, the
        # model that judged it, and the libraries that split the sentences.
        params = dataclasses.asdict(cfg)
        params["boundary_stats"] = dict(chunker.boundary_stats)
        params["document"] = _document_stamp(self._file_path, text)
        params["llm"] = _llm_stamp(self._client)
        params["environment"] = _environment_stamp()
        self._cache.save(self._file_path, chunks, enriched=False,
                         variant=key, params=params)
        return chunks

    @staticmethod
    def _report_anchor_loss(chunks: list[str], qa_pairs: list[dict]) -> int:
        #reports on low filter removal
        blob = " ".join(" ".join(c.split()) for c in chunks)
        lost = [q for q in qa_pairs if " ".join(q["source_text"].split()) not in blob]
        n = len(qa_pairs) or 1
        print(f"[anchors] {len(qa_pairs) - len(lost)}/{len(qa_pairs)} anchors present in the "
              f"LLM chunks — {len(lost)} lost ({len(lost) * 100 // n} %)")
        for q in lost[:3]:
            print(f"          fehlt: {q['question'][:70]}")
        return len(lost)

    def _evaluate(self, chunk_sets: ChunkSets, parent_child: ParentChild,
                  qa_pairs: list[dict], store: "VectorStore") -> list[dict]:
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


# --------------------------------------------------------------- provenance

def _document_stamp(path: str, text: str) -> dict:
    """The cache is named after the file stem; this ties it to the content."""
    return {
        "path": str(path),
        "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "chars": len(text),
    }


def _llm_stamp(client: OllamaClient) -> dict:
    """Decoder settings plus the identity of the model behind the tag.

    The tag `qwen3.5:4b` can point to different weights after a re-pull; the
    manifest digest from /api/tags pins the exact model. A lookup failure is
    recorded, never raised: provenance must not abort a run.
    """
    info = {
        "model": client.model, "seed": client.seed, "temperature": client.temperature,
        "num_ctx": client.num_ctx, "thinking": client.thinking,
        "model_digest": None, "ollama_version": None,
    }
    try:
        with httpx.Client(timeout=5.0) as http:
            wanted = {client.model, f"{client.model}:latest"}
            models = http.get(f"{client.base_url}/api/tags").json().get("models", [])
            match = next((m for m in models
                          if m.get("name") in wanted or m.get("model") in wanted), None)
            info["model_digest"] = match.get("digest") if match else None
            info["ollama_version"] = http.get(f"{client.base_url}/api/version").json().get("version")
    except Exception as exc:  # noqa: BLE001 - any failure is just recorded
        info["lookup_error"] = f"{type(exc).__name__}: {exc}"[:200]
    return info


def _environment_stamp() -> dict:
    """Versions of everything between the file and the sentence list."""
    def version(name: str):
        try:
            return metadata.version(name)
        except metadata.PackageNotFoundError:
            return None
    return {
        "python": platform.python_version(),
        **{name: version(name) for name in ("nltk", "langdetect", "pypdf", "httpx")},
    }
