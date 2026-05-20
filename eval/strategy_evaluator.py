from __future__ import annotations

import re

import numpy as np
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter

from llm_chunker.vectorstore import VectorStore


def build_strategies(text: str) -> dict[str, list[str]]:
    """Build non-ML baseline chunk sets from the given text."""
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
    """
    Semantic chunking using the already-loaded VectorStore embedding function.
    Splits text into sentences, embeds them, then breaks at high cosine-distance
    positions (75th percentile threshold). No second model load needed.
    """
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if len(s.strip()) > 20]
    if len(sentences) < 2:
        return [text]

    embeddings = np.array(embedding_fn(sentences))
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normed = embeddings / np.maximum(norms, 1e-9)
    distances = 1 - (normed[:-1] * normed[1:]).sum(axis=1)

    threshold = np.percentile(distances, 75)
    breakpoints = [i for i, d in enumerate(distances) if d > threshold]

    chunks, start = [], 0
    for bp in breakpoints:
        chunk = " ".join(sentences[start:bp + 1])
        if chunk:
            chunks.append(chunk)
        start = bp + 1
    tail = " ".join(sentences[start:])
    if tail:
        chunks.append(tail)

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

        hits, reciprocal_ranks, top1_distances = 0, [], []

        for i, qa in enumerate(qa_pairs, 1):
            if i % 10 == 0 or i == 1:
                print(f"    [{i}/{len(qa_pairs)}] querying...", flush=True)
            results = self._store.query(collection, qa["question"], k=self._k)
            top1_distances.append(results[0].distance if results else 1.0)

            rank = next(
                (j for j, r in enumerate(results, 1) if self._is_hit(qa["source_text"], r.chunk_text)),
                None,
            )
            if rank is not None:
                hits += 1
                reciprocal_ranks.append(1.0 / rank)
            else:
                reciprocal_ranks.append(0.0)

        n = len(qa_pairs)
        if n == 0:
            print(f"  [eval] No questions available for '{name}' — skipping metrics.")
            return {
                "strategy":           name,
                "chunk_count":        len(chunks),
                "avg_chunk_len":      0,
                f"hit_rate@{self._k}": 0.0,
                "mrr":                0.0,
                "avg_dist_top1":      0.0,
            }
        return {
            "strategy":           name,
            "chunk_count":        len(chunks),
            "avg_chunk_len":      round(sum(len(c) for c in chunks) / max(1, len(chunks))),
            f"hit_rate@{self._k}": round(hits / n * 100, 1),
            "mrr":                round(sum(reciprocal_ranks) / n, 3),
            "avg_dist_top1":      round(sum(top1_distances) / n, 3),
        }

    @staticmethod
    def _is_hit(source: str, retrieved: str, window: int = 40) -> bool:
        """Check if the retrieved chunk contains content from the source.

        Slides a window over the *retrieved* chunk and checks how many windows
        appear in the source. This is fair regardless of chunk size: a small
        fixed_256 chunk that covers the relevant passage scores just as well as
        a large LLM chunk that covers it.
        """
        if not source or not retrieved:
            return False
        a = " ".join(source.split())
        b = " ".join(retrieved.split())
        if len(b) < window:
            return b in a
        matches = sum(1 for i in range(0, len(b) - window + 1, window) if b[i:i + window] in a)
        return matches >= 1
