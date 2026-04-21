"""
VectorStore — ChromaDB wrapper for chunking evaluation.

Stores chunks as vectors and retrieves the most similar ones for a query.

Usage:
    store = VectorStore()
    store.add_chunks("llm_semantic", ["chunk1 text", "chunk2 text", ...])
    results = store.query("llm_semantic", "How does CRISPR work?", k=3)
"""

from __future__ import annotations

import chromadb
from chromadb.utils import embedding_functions
from dataclasses import dataclass


@dataclass
class RetrievalResult:
    """A single retrieval result."""
    chunk_text: str
    distance: float        # lower = more similar (cosine distance)
    chunk_index: int
    metadata: dict


class VectorStore:
    """
    Each chunking strategy gets its own ChromaDB collection.
    Same query can be run against all collections to compare results.
    """

    def __init__(
        self,
        persist_dir: str = "./chroma_db",
        embedding_model: str = "nomic-ai/nomic-embed-text-v1.5",
    ):
        self._client = chromadb.PersistentClient(path=persist_dir)
        self._embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=embedding_model,
            trust_remote_code=True,
        )

    def add_chunks(
        self,
        collection_name: str,
        chunks: list[str],
        source: str = "",
    ) -> None:
        """Embed and store chunks in a named collection."""
        collection = self._client.get_or_create_collection(
            name=collection_name,
            embedding_function=self._embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )

        # Clear existing data
        existing = collection.count()
        if existing > 0:
            collection.delete(ids=[f"chunk_{i}" for i in range(existing)])

        ids = [f"chunk_{i}" for i in range(len(chunks))]
        metadatas = [
            {"source": source, "chunk_index": i, "char_count": len(c), "original_text": c}
            for i, c in enumerate(chunks)
        ]
        prefixed_chunks = [f"search_document: {c}" for c in chunks]

        collection.add(documents=prefixed_chunks, ids=ids, metadatas=metadatas)
        print(f"[vectorstore] '{collection_name}': {len(chunks)} chunks stored")

    def query(
        self,
        collection_name: str,
        query_text: str,
        k: int = 3,
    ) -> list[RetrievalResult]:
        """Find the k most similar chunks to the query."""
        collection = self._client.get_collection(
            name=collection_name,
            embedding_function=self._embedding_fn,
        )

        results = collection.query(
            query_texts=[f"search_query: {query_text}"],
            n_results=min(k, collection.count()),
        )

        return [
            RetrievalResult(
                chunk_text=results["metadatas"][0][i].get("original_text", results["documents"][0][i]),
                distance=results["distances"][0][i],
                chunk_index=results["metadatas"][0][i]["chunk_index"],
                metadata=results["metadatas"][0][i],
            )
            for i in range(len(results["documents"][0]))
        ]

    def list_collections(self) -> list[str]:
        return [c.name for c in self._client.list_collections()]

    def delete_collection(self, name: str) -> None:
        self._client.delete_collection(name)