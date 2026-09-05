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
    #Calls Ollama's /api/embed endpoint

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
            response = self._client.post(
                f"{self._base_url}/api/embed",
                json={"model": self._model, "input": batch},
            )
            if response.status_code != 200:
                raise RuntimeError(
                    f"Ollama embed failed ({response.status_code}) for batch "
                    f"[{start}:{start + len(batch)}]: {response.text[:300]}"
                )
            embeddings.extend(response.json()["embeddings"])
        return embeddings


class VectorStore:

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
        """Speichert `chunks` (das, was EINGEBETTET wird).

        `display_texts` entkoppelt davon, was beim Retrieval ZURUECKGEGEBEN wird —
        die Grundlage fuer Parent-Child: eingebettet werden die kleinen Children,
        zurueckgegeben wird der zugehoerige grosse Parent.
        """
        # Collection komplett verwerfen statt Eintraege per ID zu loeschen.
        # Grund: Beim Wiederbefuellen mit deutlich WENIGER Chunks (z.B. 3447 NASA
        # -> 152 stoic) blieb der HNSW-Index degradiert zurueck und lieferte
        # unbrauchbare Nachbarn (stoic/fixed_256: 14 % statt 78 % Hit@1).
        # Ein frisch angelegter Index ist die einzige verlaessliche Variante.
        try:
            self._client.delete_collection(name=collection_name)
        except Exception:
            pass  # existierte noch nicht

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
        print(f"[vectorstore] '{collection_name}': {len(chunks)} chunks stored")

    _DEDUPE_OVERFETCH = 6      # wie viele Kandidaten je gewuenschtem Treffer geholt werden

    def query(
        self,
        collection_name: str,
        query_text: str,
        k: int = 3,
        dedupe_by_text: bool = False,
    ) -> list[RetrievalResult]:
        """Sucht die k aehnlichsten Eintraege.

        dedupe_by_text: Treffer, die denselben Text zurueckgeben, zaehlen nur
        einmal. Noetig fuer Parent-Child — dort zeigen oft mehrere Children auf
        denselben Parent, der sonst k Plaetze belegen wuerde. Es wird ueberholt
        und anschliessend auf k reduziert.
        """
        collection = self._client.get_collection(
            name=collection_name,
            embedding_function=self._embedding_fn,
        )

        wanted = k * self._DEDUPE_OVERFETCH if dedupe_by_text else k
        results = collection.query(
            query_texts=[query_text],
            n_results=min(wanted, collection.count()),
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
