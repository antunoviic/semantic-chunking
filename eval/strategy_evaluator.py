from __future__ import annotations

try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker

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


def build_semantic_lc(text: str) -> list[str]:
    """
    Build LangChain semantic chunks (loads HuggingFace model).
    Called separately so the model is released before VectorStore loads its own instance.
    """
    import gc
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    chunks = SemanticChunker(embeddings).split_text(text)
    del embeddings
    gc.collect()  # force release before VectorStore loads the same model
    print("  [semantic_lc] Model released from memory.", flush=True)
    return chunks


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
    def _is_hit(source: str, retrieved: str, threshold: float = 0.3) -> bool:
        if not source:
            return False
        window = 60
        a = " ".join(source.split())
        b = " ".join(retrieved.split())
        matches = sum(1 for i in range(0, len(a) - window + 1, window) if a[i:i + window] in b)
        return (matches / max(1, len(a) // window)) >= threshold
