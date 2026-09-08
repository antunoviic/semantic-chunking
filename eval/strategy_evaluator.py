from __future__ import annotations

import re
from difflib import SequenceMatcher

from langchain_core.embeddings import Embeddings
from langchain_experimental.text_splitter import SemanticChunker
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter

from llm_chunker.vectorstore import VectorStore


class _OllamaLCEmbeddings(Embeddings):
    #Adapter: langcain uses same embedding as the semantic chunker

    def __init__(self, embedding_fn) -> None:
        self._fn = embedding_fn

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._fn(list(texts))

    def embed_query(self, text: str) -> list[float]:
        return self._fn([text])[0]


def _avg_len(chunks: list[str]) -> int:
    return round(sum(map(len, chunks)) / max(1, len(chunks)))


def build_strategies(text: str, match_len: int | None = None) -> dict[str, list[str]]:

    def fixed(size: int, overlap: int) -> list[str]:
        return CharacterTextSplitter(chunk_size=size, chunk_overlap=overlap, separator=" ").split_text(text)

    def recursive(size: int = 512, overlap: int = 50) -> list[str]:
        return RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=overlap).split_text(text)

    out = {
        "fixed_256": fixed(256, 20),
        "fixed_512": fixed(512, 50),
        "recursive": recursive(),
    }
    if match_len: #matches length of llm-chunker to show difference in chunkking not length
        out[f"fixed_matched_{match_len}"] = fixed(match_len, 50)
        size, best = match_len, None
        for _ in range(6):
            cand = recursive(size, 50)
            got = _avg_len(cand)
            if best is None or abs(got - match_len) < abs(_avg_len(best) - match_len):
                best = cand
            if abs(got - match_len) / match_len <= 0.04:
                break
            size = max(60, round(size * match_len / max(1, got)))
        out[f"recursive_matched_{match_len}"] = best
    return out


def build_semantic_lc(text: str, embedding_fn) -> list[str]:

    chunker = SemanticChunker(
        _OllamaLCEmbeddings(embedding_fn),
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=95,
    )
    chunks = chunker.split_text(text)
    return chunks or [text]

#splits parents into children for search
#search through children, retireving parent
def build_parent_child(
    parents: list[str],
    child_chars: int = 250,
    child_overlap: int = 30,
) -> tuple[list[str], list[str]]:

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=child_chars, chunk_overlap=child_overlap
    )
    children: list[str] = []
    parent_of: list[str] = []
    for parent in parents:
        subs = [s for s in splitter.split_text(parent) if s.strip()] or [parent]
        for sub in subs:
            children.append(sub)
            parent_of.append(parent)
    return children, parent_of


class StrategyEvaluator:

    REPORT_KS = (1, 3, 5, 10)

    def __init__(self, store: VectorStore, k: int = 3) -> None:
        self._store = store
        self._k = k

    def evaluate(
        self,
        name: str,
        chunks: list[str],
        qa_pairs: list[dict],
        display_texts: list[str] | None = None,
    ) -> dict:
        collection = f"eval_{name}"
        dedupe = display_texts is not None
        # until hit@10
        fetch_k = max(self._k, max(self.REPORT_KS))
        print(f"    Adding {len(chunks)} chunks to vector store...", flush=True)
        self._store.add_chunks(collection, chunks, source=name, display_texts=display_texts)
        print(f"    Running {len(qa_pairs)} queries...", flush=True)

        hits_at = {kk: 0 for kk in self.REPORT_KS}
        reciprocal_ranks, top1_distances, retrieved_chars = [], [], []
        missed_questions = []
        question_ranks: list[dict] = []

        for i, qa in enumerate(qa_pairs, 1):
            if i % 10 == 0 or i == 1:
                print(f"    [{i}/{len(qa_pairs)}] querying...", flush=True)
            results = self._store.query(collection, qa["question"], k=fetch_k,
                                        dedupe_by_text=dedupe)
            top1_distances.append(results[0].distance if results else 1.0)
            # context costs 
            retrieved_chars.append(sum(len(r.chunk_text) for r in results[:self._k]))

            rank = next(
                (j for j, r in enumerate(results, 1) if self._is_hit(qa["source_text"], r.chunk_text)),
                None,
            )
            question_ranks.append({"question": qa["question"], "rank": rank})
            if rank is not None:
                for kk in self.REPORT_KS:
                    if rank <= kk:
                        hits_at[kk] += 1
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
                **{f"hit_rate@{kk}": 0.0 for kk in self.REPORT_KS},
                "mrr":                 0.0,
                "avg_dist_top1":       0.0,
                "avg_retrieved_chars": 0,
            }
        return {
            "strategy":            name,
            "chunk_count":         len(chunks),
            "avg_chunk_len":       round(
                sum(len(c) for c in (set(display_texts) if display_texts else chunks))
                / max(1, len(set(display_texts)) if display_texts else len(chunks))
            ),
            **{f"hit_rate@{kk}": round(hits_at[kk] / n * 100, 1) for kk in self.REPORT_KS},
            "mrr":                 round(sum(reciprocal_ranks) / n, 3),
            "avg_dist_top1":       round(sum(top1_distances) / n, 3),
            # Context cost: how many characters land in the LLM prompt per query
            # High hit rates achieved with huge chunks are easier to get becasuse right chunk will be in it
            "avg_retrieved_chars": round(sum(retrieved_chars) / n),
            # Failed questions with what was wrongly retrieved — for error analysis
            "missed_questions":    missed_questions,
            #average so that isn't just dependant whether its in top10 but also if it wins for top 1/3 etc.
            "question_ranks":      question_ranks,
        }

    _TOPIC_PREFIX = re.compile(r"^\[Topic:[^\]]*\]\s*")
    COVERAGE_THRESHOLD = 0.8

    @classmethod
    def coverage(cls, source: str, retrieved: str) -> float:
        #what portion of substring is in the chunk 0.1
        if not source or not retrieved:
            return 0.0
        a = " ".join(source.split())
        b = " ".join(cls._TOPIC_PREFIX.sub("", retrieved.strip()).split())
        if not a:
            return 0.0
        if a in b:                     
            return 1.0
        match = SequenceMatcher(None, a, b, autojunk=False).find_longest_match(0, len(a), 0, len(b))
        return match.size / len(a)

    @classmethod
    def _is_hit(cls, source: str, retrieved: str) -> bool:
        #hit when threshold met
        return cls.coverage(source, retrieved) >= cls.COVERAGE_THRESHOLD

    @classmethod
    def _is_hit_legacy(cls, source: str, retrieved: str, window: int = 40) -> bool:
        #old criteria, hit if length containment is met
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
