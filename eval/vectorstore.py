from __future__ import annotations

import os
from dataclasses import dataclass

try:
    import chromadb
except ModuleNotFoundError as exc:                  # pragma: no cover
    raise ModuleNotFoundError(
        "eval.vectorstore needs ChromaDB: pip install chromadb. "
        "Chunking itself does not require it."
    ) from exc
import httpx
from chromadb import EmbeddingFunction, Embeddings, Documents
from llm_semantic_chunker._logging import get_logger
from llm_semantic_chunker._retry import post_with_retries

logger = get_logger(__name__)


@dataclass
class RetrievalResult:
    """One retrieved chunk with the distance that ranked it."""

    chunk_text: str
    distance: float        # lower = more similar (cosine distance)
    chunk_index: int
    metadata: dict


class OllamaEmbeddingFunction(EmbeddingFunction[Documents]):
    """Embeds text through a local Ollama model, in ChromaDB's interface.

    Keeping the embedder behind Ollama rather than loading a model in-process
    is what lets the evaluation run on 8 GB of memory: the weights live in one
    server shared by every arm instead of in each Python process.
    """


    def __init__(
        self,
        model: str = "bge-m3",
        base_url: str = "",
    ) -> None:
        self._model = model
        self._base_url = (base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")).rstrip("/")
        self._client = httpx.Client(timeout=httpx.Timeout(connect=10.0, read=120.0, write=10.0, pool=10.0))

    # bge-m3 context window is ~8192 tokens;
    _MAX_CHARS = 5000
    _BATCH_SIZE = 50

    def __call__(self, input: Documents) -> Embeddings:
        truncated = [t[:self._MAX_CHARS] or " " for t in input]
        embeddings: Embeddings = []
        for start in range(0, len(truncated), self._BATCH_SIZE):
            batch = truncated[start:start + self._BATCH_SIZE]
            # Same policy as the chat client: a runner restart or a transient 5xx
            # must not abort a 40-minute evaluation; a 4xx is our mistake.
            try:
                response = post_with_retries(
                    self._client, f"{self._base_url}/api/embed",
                    json={"model": self._model, "input": batch},
                )
            except httpx.HTTPStatusError as exc:
                raise RuntimeError(
                    f"Ollama embed failed ({exc.response.status_code}) for batch "
                    f"[{start}:{start + len(batch)}]: {exc.response.text[:300]}"
                ) from exc
            embeddings.extend(response.json()["embeddings"])
        return embeddings


class VectorStore:
    """A ChromaDB collection per arm, rebuilt on every run.

    Belongs to the evaluation rather than to chunking, and is not exported from
    the package. `add_chunks` deletes an existing collection before writing, so
    a run never reads vectors left over from an earlier one. The cost is that
    the HNSW graph is rebuilt each time, and its construction is not seeded —
    identical inputs can therefore give slightly different rankings between
    runs.
    """

    #each chunking strategy gets its own chromaDB collection


    def __init__(
        self,
        persist_dir: str = "./chroma_db",
        embedding_model: str = "bge-m3",
    ):
        self._client = chromadb.PersistentClient(path=persist_dir)
        self._embedding_fn = OllamaEmbeddingFunction(model=embedding_model)

    def add_chunks(
        self,
        collection_name: str,
        chunks: list[str],
        source: str = "",
        display_texts: list[str] | None = None,
    ) -> None:
    #embed children
    #retrieve parents

        # delete collection after each run to ensure consistency
        try:
            self._client.delete_collection(name=collection_name)
        except Exception:
            pass  # did not exist yet

        collection = self._client.create_collection(
            name=collection_name,
            embedding_function=self._embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )

        ids = [f"chunk_{i}" for i in range(len(chunks))]
        shown = display_texts if display_texts is not None else chunks
        if len(shown) != len(chunks):
            raise ValueError("display_texts must align 1:1 with chunks")
        metadatas = [
            {"source": source, "chunk_index": i, "char_count": len(c), "original_text": shown[i]}
            for i, c in enumerate(chunks)
        ]
        batch_size = 50
        for start in range(0, len(chunks), batch_size):
            end = start + batch_size
            collection.add(
                documents=chunks[start:end],
                ids=ids[start:end],
                metadatas=metadatas[start:end],
            )
        logger.info(f"[vectorstore] '{collection_name}': {len(chunks)} chunks stored")

    _DEDUPE_OVERFETCH = 6      # candidates for each hit

    def query(
        self,
        collection_name: str,
        query_text: str,
        k: int = 3,
        dedupe_by_text: bool = False,
    ) -> list[RetrievalResult]:
        collection = self._client.get_collection(
            name=collection_name,
            embedding_function=self._embedding_fn,
        )

        wanted = k * self._DEDUPE_OVERFETCH if dedupe_by_text else k
        count = collection.count()
        if count == 0:                 # ChromaDB rejects n_results=0
            return []
        results = collection.query(
            query_texts=[query_text],
            n_results=min(wanted, count),
        )

        out: list[RetrievalResult] = []
        seen: set[str] = set()
        for i in range(len(results["documents"][0])):
            meta = results["metadatas"][0][i]
            text = meta.get("original_text", results["documents"][0][i])
            if dedupe_by_text:
                if text in seen:
                    continue
                seen.add(text)
            out.append(
                RetrievalResult(
                    chunk_text=text,
                    distance=results["distances"][0][i],
                    chunk_index=meta["chunk_index"],
                    metadata=meta,
                )
            )
            if len(out) >= k:
                break
        return out

    def list_collections(self) -> list[str]:
        return [c.name for c in self._client.list_collections()]

    def delete_collection(self, name: str) -> None:
        self._client.delete_collection(name)
