"""
VectorStore — ChromaDB wrapper for chunking evaluation.

Embeddings are computed via Ollama (nomic-embed-text) so no PyTorch/ONNX
is loaded inside the Python process.

Usage:
    store = VectorStore()
    store.add_chunks("llm_semantic", ["chunk1 text", "chunk2 text", ...])
    results = store.query("llm_semantic", "How does CRISPR work?", k=3)
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import chromadb
import httpx
from chromadb import EmbeddingFunction, Embeddings, Documents


@dataclass
class RetrievalResult:
    chunk_text: str
    distance: float        # lower = more similar (cosine distance)
    chunk_index: int
    metadata: dict


class OllamaEmbeddingFunction(EmbeddingFunction[Documents]):
    """Calls Ollama's /api/embed endpoint — no native ML in this process."""

    def __init__(
        self,
        model: str = "nomic-embed-text",
        base_url: str = "",
    ) -> None:
        self._model = model
        self._base_url = (base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")).rstrip("/")
        self._client = httpx.Client(timeout=httpx.Timeout(connect=10.0, read=120.0, write=10.0, pool=10.0))

    # nomic-embed-text context window is ~8192 tokens; truncate to ~6000 chars to stay safe
    _MAX_CHARS = 5000

    def __call__(self, input: Documents) -> Embeddings:
        truncated = [t[:self._MAX_CHARS] for t in input]
        response = self._client.post(
            f"{self._base_url}/api/embed",
            json={"model": self._model, "input": truncated},
        )
        response.raise_for_status()
        return response.json()["embeddings"]


class VectorStore:
    """
    Each chunking strategy gets its own ChromaDB collection.
    Same query can be run against all collections to compare results.
    """

    def __init__(
        self,
        persist_dir: str = "./chroma_db",
        embedding_model: str = "nomic-embed-text",
    ):
        self._client = chromadb.PersistentClient(path=persist_dir)
        self._embedding_fn = OllamaEmbeddingFunction(model=embedding_model)

    def add_chunks(
        self,
        collection_name: str,
        chunks: list[str],
        source: str = "",
    ) -> None:
        collection = self._client.get_or_create_collection(
            name=collection_name,
            embedding_function=self._embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )

        existing = collection.count()
        if existing > 0:
            collection.delete(ids=[f"chunk_{i}" for i in range(existing)])

        ids = [f"chunk_{i}" for i in range(len(chunks))]
        metadatas = [
            {"source": source, "chunk_index": i, "char_count": len(c), "original_text": c}
            for i, c in enumerate(chunks)
        ]
        prefixed = [f"search_document: {c}" for c in chunks]

        batch_size = 50
        for start in range(0, len(chunks), batch_size):
            end = start + batch_size
            collection.add(
                documents=prefixed[start:end],
                ids=ids[start:end],
                metadatas=metadatas[start:end],
            )
        print(f"[vectorstore] '{collection_name}': {len(chunks)} chunks stored")

    def query(
        self,
        collection_name: str,
        query_text: str,
        k: int = 3,
    ) -> list[RetrievalResult]:
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
