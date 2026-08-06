from __future__ import annotations

import re

from langchain_core.embeddings import Embeddings
from langchain_experimental.text_splitter import SemanticChunker
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter

from llm_chunker.vectorstore import VectorStore


class _OllamaLCEmbeddings(Embeddings):
    """Adapter: expose the project's Ollama (bge-m3) embedding function as a
    LangChain Embeddings object, so LangChain's SemanticChunker uses the very
    same embedder as retrieval (fair comparison, no extra model load)."""

    def __init__(self, embedding_fn) -> None:
        self._fn = embedding_fn

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._fn(list(texts))

    def embed_query(self, text: str) -> list[float]:
        return self._fn([text])[0]


def build_strategies(text: str) -> dict[str, list[str]]:
    #baseline strategies
    def fixed(size: int, overlap: int) -> list[str]:
        return CharacterTextSplitter(chunk_size=size, chunk_overlap=overlap, separator=" ").split_text(text)

    def recursive(size: int = 512, overlap: int = 50) -> list[str]:
        return RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=overlap).split_text(text)

    return {
        "fixed_256": fixed(256, 20),
        "fixed_512": fixed(512, 50),
        "recursive": recursive(),
    }


def build_semantic_lc(text: str, embedding_fn) -> list[str]:
    """Semantic chunking via LangChain's SemanticChunker (state-of-the-art baseline).

    Uses LangChain's actual SemanticChunker, fed with the project's bge-m3 Ollama
    embeddings through _OllamaLCEmbeddings so the baseline shares the same embedder
    as retrieval. A boundary is placed where the cosine distance between consecutive
    sentences exceeds the 75th percentile (LangChain's own default is 95).
    """
    chunker = SemanticChunker(
        _OllamaLCEmbeddings(embedding_fn),
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=75,
    )
    chunks = chunker.split_text(text)
    return chunks or [text]


class StrategyEvaluator:
    """SRP: evaluates one chunking strategy and returns metrics."""

    def __init__(self, store: VectorStore, k: int = 3) -> None:
        self._store = store
        self._k = k

    def evaluate(self, name: str, chunks: list[str], qa_pairs: list[dict]) -> dict:
        collection = f"eval_{name}"
        print(f"    Adding {len(chunks)} chunks to vector store...", flush=True)
        self._store.add_chunks(collection, chunks, source=name)
        print(f"    Running {len(qa_pairs)} queries...", flush=True)

        hits, hits_top1, reciprocal_ranks, top1_distances, retrieved_chars = 0, 0, [], [], []
        missed_questions = []

        for i, qa in enumerate(qa_pairs, 1):
            if i % 10 == 0 or i == 1:
                print(f"    [{i}/{len(qa_pairs)}] querying...", flush=True)
            results = self._store.query(collection, qa["question"], k=self._k)
            top1_distances.append(results[0].distance if results else 1.0)
            retrieved_chars.append(sum(len(r.chunk_text) for r in results))

            rank = next(
                (j for j, r in enumerate(results, 1) if self._is_hit(qa["source_text"], r.chunk_text)),
                None,
            )
            if rank is not None:
                hits += 1
                if rank == 1:
                    hits_top1 += 1
                reciprocal_ranks.append(1.0 / rank)
            else:
                reciprocal_ranks.append(0.0)
                missed_questions.append({
                    "question":      qa["question"],
                    "source_text":   qa["source_text"][:200],
                    "top1_distance": round(results[0].distance, 3) if results else None,
                    "top1_preview":  results[0].chunk_text[:200] if results else "",
                })

        n = len(qa_pairs)
        if n == 0:
            print(f"  [eval] No questions available for '{name}' — skipping metrics.")
            return {
                "strategy":            name,
                "chunk_count":         len(chunks),
                "avg_chunk_len":       0,
                "hit_rate@1":          0.0,
                f"hit_rate@{self._k}": 0.0,
                "mrr":                 0.0,
                "avg_dist_top1":       0.0,
                "avg_retrieved_chars": 0,
            }
        return {
            "strategy":            name,
            "chunk_count":         len(chunks),
            "avg_chunk_len":       round(sum(len(c) for c in chunks) / max(1, len(chunks))),
            "hit_rate@1":          round(hits_top1 / n * 100, 1),
            f"hit_rate@{self._k}": round(hits / n * 100, 1),
            "mrr":                 round(sum(reciprocal_ranks) / n, 3),
            "avg_dist_top1":       round(sum(top1_distances) / n, 3),
            # Context cost: how many characters land in the LLM prompt per query.
            # High hit rates achieved with huge chunks are cheaper to fake.
            "avg_retrieved_chars": round(sum(retrieved_chars) / n),
            # Failed questions with what was wrongly retrieved — for error analysis
            "missed_questions":    missed_questions,
        }

    _TOPIC_PREFIX = re.compile(r"^\[Topic:[^\]]*\]\s*")

    @classmethod
    def _is_hit(cls, source: str, retrieved: str, window: int = 40) -> bool:
        """Check if the retrieved chunk contains content from the source.

        The [Topic: ...] enrichment prefix is stripped first so enriched
        chunks are judged on the same text as their raw counterparts.
        A full-containment check handles the common case (chunk covers the
        whole source sentence); the sliding window keeps it fair for small
        chunks that only cover part of the sentence.
        """
        if not source or not retrieved:
            return False
        retrieved = cls._TOPIC_PREFIX.sub("", retrieved.strip())
        a = " ".join(source.split())
        b = " ".join(retrieved.split())
        if a in b:
            return True
        if len(b) < window:
            return b in a
        matches = sum(1 for i in range(0, len(b) - window + 1, window) if b[i:i + window] in a)
        return matches >= 1
