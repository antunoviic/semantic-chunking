"""
Application pipeline layer.

Provides focused classes (SRP) that the ChunkingPipeline orchestrates:
  PDFReader          — extract text from a PDF
  ChunkCache         — load / save chunk JSON cache
  QuestionGenerator  — generate QA pairs via Qwen
  StrategyEvaluator  — evaluate one chunking strategy
  ResultReporter     — print table, save JSON + charts
  ChunkingPipeline   — orchestrate the full run (DIP: depends on abstractions)
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Optional

from pypdf import PdfReader
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings

from llm_chunker import LLMChunker, QwenClient
from llm_chunker.interfaces import LLMClient
from llm_chunker.vectorstore import VectorStore


# ── Chunking strategies registry ───────────────────────────────────────────────

def _build_strategies(text: str) -> dict[str, list[str]]:
    """Build all baseline chunk sets from the given text."""
    def fixed(size: int, overlap: int) -> list[str]:
        return CharacterTextSplitter(chunk_size=size, chunk_overlap=overlap, separator=" ").split_text(text)

    def recursive(size: int = 512, overlap: int = 50) -> list[str]:
        return RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=overlap).split_text(text)

    def semantic_lc() -> list[str]:
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        return SemanticChunker(embeddings).split_text(text)

    return {
        "fixed_256":   fixed(256, 20),
        "fixed_512":   fixed(512, 50),
        "recursive":   recursive(),
        "semantic_lc": semantic_lc(),
    }


# ── PDFReader ──────────────────────────────────────────────────────────────────

class PDFReader:
    """SRP: reads a PDF file and returns its plain text."""

    def __init__(self, pdf_path: str) -> None:
        self._path = Path(pdf_path)
        self._reader = PdfReader(str(self._path))

    @property
    def page_count(self) -> int:
        return len(self._reader.pages)

    def extract_text(self) -> str:
        pages = [page.extract_text() or "" for page in self._reader.pages]
        return "\n\n".join(p.strip() for p in pages if p.strip())


# ── ChunkCache ─────────────────────────────────────────────────────────────────

class ChunkCache:
    """SRP: loads and saves chunk JSON caches with automatic backup."""

    def __init__(self, cache_dir: Path = Path("./chunks_cache")) -> None:
        self._dir = cache_dir

    def _path(self, pdf_path: str, enriched: bool) -> Path:
        suffix = "_enriched" if enriched else ""
        return self._dir / (Path(pdf_path).stem + suffix + ".json")

    def load(self, pdf_path: str, enriched: bool = False) -> Optional[list[str]]:
        path = self._path(pdf_path, enriched)
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        print(f"[cache] Loaded {len(data['chunks'])} chunks from {path.name}")
        print(f"        Chunked on: {data['chunked_at']}")
        return data["chunks"]

    def save(self, pdf_path: str, chunks: list[str], enriched: bool = False) -> None:
        self._dir.mkdir(exist_ok=True)
        path = self._path(pdf_path, enriched)
        if path.exists():
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup = path.with_suffix(f".{ts}.json")
            path.rename(backup)
            print(f"[cache] Old chunks backed up to {backup.name}")
        data = {
            "source":      Path(pdf_path).name,
            "chunked_at":  datetime.now().isoformat(),
            "chunk_count": len(chunks),
            "chunks":      chunks,
        }
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[cache] Saved {len(chunks)} chunks to {path.name}")


# ── QuestionGenerator ──────────────────────────────────────────────────────────

class QuestionGenerator:
    """SRP: generates QA pairs from chunks via an LLM, with caching."""

    def __init__(self, client: LLMClient, cache_dir: Path = Path("./eval_cache")) -> None:
        self._client = client
        self._dir = cache_dir

    def generate(
        self,
        chunks: list[str],
        pdf_stem: str,
        max_questions: int = 50,
        regen: bool = False,
    ) -> list[dict]:
        self._dir.mkdir(exist_ok=True)
        qa_path = self._dir / f"{pdf_stem}_questions.json"

        if qa_path.exists() and not regen:
            data = json.loads(qa_path.read_text(encoding="utf-8"))
            print(f"[questions] Loaded {len(data)} questions from cache")
            return data

        if qa_path.exists() and regen:
            qa_path.unlink()

        step = max(1, len(chunks) // max_questions)
        sampled = [(i, chunks[i]) for i in range(0, len(chunks), step)][:max_questions]
        print(f"[questions] Generating {len(sampled)} questions via Qwen...")

        qa_pairs = []
        for idx, (chunk_idx, chunk_text) in enumerate(sampled):
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You generate exactly one specific question that is answered "
                        "by the given text. Output only the question, nothing else."
                    ),
                },
                {"role": "user", "content": chunk_text},
            ]
            question = self._client.chat(messages).strip()
            question = re.sub(r"^(Question:|Q:)\s*", "", question, flags=re.IGNORECASE).strip()
            qa_pairs.append({
                "question":    question,
                "source_text": chunk_text,
                "chunk_index": chunk_idx,
            })
            print(f"  [{idx+1}/{len(sampled)}] {question[:80]}")

        qa_path.write_text(json.dumps(qa_pairs, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[questions] Saved to {qa_path}")
        return qa_pairs


# ── StrategyEvaluator ──────────────────────────────────────────────────────────

class StrategyEvaluator:
    """SRP: evaluates one chunking strategy and returns metrics."""

    def __init__(self, store: VectorStore, k: int = 3) -> None:
        self._store = store
        self._k = k

    def evaluate(self, name: str, chunks: list[str], qa_pairs: list[dict]) -> dict:
        collection = f"eval_{name}"
        self._store.add_chunks(collection, chunks, source=name)

        hits, reciprocal_ranks, top1_distances = 0, [], []

        for qa in qa_pairs:
            results = self._store.query(collection, qa["question"], k=self._k)
            top1_distances.append(results[0].distance if results else 1.0)

            rank = next(
                (i for i, r in enumerate(results, 1) if self._is_hit(qa["source_text"], r.chunk_text)),
                None,
            )
            if rank is not None:
                hits += 1
                reciprocal_ranks.append(1.0 / rank)
            else:
                reciprocal_ranks.append(0.0)

        n = len(qa_pairs)
        return {
            "strategy":           name,
            "chunk_count":        len(chunks),
            "avg_chunk_len":      round(sum(len(c) for c in chunks) / max(1, len(chunks))),
            f"hit_rate@{self._k}": round(hits / n * 100, 1),
            "mrr":                round(sum(reciprocal_ranks) / n, 3),
            "avg_dist_top1":      round(sum(top1_distances) / n, 3),
        }

    @staticmethod
    def _is_hit(source: str, retrieved: str, threshold: float = 0.3) -> bool:
        if not source:
            return False
        window = 60
        a = " ".join(source.split())
        b = " ".join(retrieved.split())
        matches = sum(1 for i in range(0, len(a) - window + 1, window) if a[i:i + window] in b)
        return (matches / max(1, len(a) // window)) >= threshold


# ── ResultReporter ─────────────────────────────────────────────────────────────

class ResultReporter:
    """SRP: prints the results table and saves JSON + chart files."""

    def __init__(self, results_dir: Path = Path("./eval_results")) -> None:
        self._dir = results_dir

    def print_table(self, results: list[dict], k: int) -> None:
        hr_key = f"hit_rate@{k}"
        header = f"{'Strategy':<15} {'Chunks':>7} {'Avg Len':>8} {f'Hit@{k}':>8} {'MRR':>7} {'Avg Dist':>9}"
        sep = "=" * len(header)
        print(f"\n{sep}\n{header}\n{sep}")
        for r in results:
            print(
                f"{r['strategy']:<15} "
                f"{r['chunk_count']:>7} "
                f"{r['avg_chunk_len']:>8} "
                f"{r[hr_key]:>7.1f}% "
                f"{r['mrr']:>7.3f} "
                f"{r['avg_dist_top1']:>9.3f}"
            )
        print(sep)

    def save_json(self, results: list[dict], pdf_stem: str) -> None:
        self._dir.mkdir(exist_ok=True)
        ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = self._dir / f"{pdf_stem}_{ts}.json"
        out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[results] Saved to {out}")

    def save_charts(self, results: list[dict], k: int, pdf_stem: str) -> None:
        try:
            import matplotlib
            import matplotlib.pyplot as plt
            matplotlib.use("Agg")
        except ImportError:
            print("[charts] matplotlib not installed — skipping")
            return

        self._dir.mkdir(exist_ok=True)
        hr_key = f"hit_rate@{k}"
        names  = [r["strategy"] for r in results]
        colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2", "#937860"]

        fig, axes = plt.subplots(1, 3, figsize=(14, 5))
        fig.suptitle(f"Chunking Strategy Comparison — {pdf_stem}", fontsize=13)

        def bar(ax, values, title, ylabel, fmt="{:.1f}"):
            bars = ax.bar(names, values, color=colors[:len(names)])
            ax.set_title(title)
            ax.set_ylabel(ylabel)
            ax.set_ylim(0, max(values) * 1.2 if max(values) > 0 else 1)
            for b, v in zip(bars, values):
                ax.text(b.get_x() + b.get_width() / 2, b.get_height() + max(values) * 0.02,
                        fmt.format(v), ha="center", va="bottom", fontsize=9)
            ax.tick_params(axis="x", rotation=20)

        bar(axes[0], [r[hr_key] for r in results],        f"Hit Rate@{k} (%)",                 "%",  "{:.1f}")
        bar(axes[1], [r["mrr"] for r in results],          "MRR (higher = better)",             "",   "{:.3f}")
        bar(axes[2], [r["avg_dist_top1"] for r in results], "Avg Distance Top-1\n(lower = better)", "", "{:.3f}")

        plt.tight_layout()
        out = self._dir / f"{pdf_stem}_comparison.png"
        plt.savefig(out, dpi=150)
        print(f"[charts] Saved to {out}")


# ── ChunkingPipeline ───────────────────────────────────────────────────────────

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
        self._pdf_path       = pdf_path
        self._rechunk        = rechunk
        self._enrich         = enrich
        self._top_k          = top_k
        self._max_questions  = max_questions
        self._regen_q        = regen_questions

        self._client  = QwenClient()
        self._cache   = ChunkCache()
        self._reader  = PDFReader(pdf_path)
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
        all_chunks = _build_strategies(text)
        all_chunks["llm"] = llm_chunks

        enriched = self._cache.load(self._pdf_path, enriched=True)
        if enriched:
            all_chunks["llm_enriched"] = enriched
        else:
            print("  [llm_enriched] No enriched cache — run with --enrich to generate.")

        for name, chunks in all_chunks.items():
            print(f"  {name:<15}: {len(chunks)} chunks")

        # 5. Evaluate
        store     = VectorStore(persist_dir="./chroma_db_eval")
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
            if cached:
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
