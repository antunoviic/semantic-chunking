from __future__ import annotations

import re
from difflib import SequenceMatcher

from langchain_core.embeddings import Embeddings
from langchain_experimental.text_splitter import SemanticChunker
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter

from eval.vectorstore import VectorStore
from llm_semantic_chunker import LLMChunker
from llm_semantic_chunker.text_splitter import TextSplitter


class _CachedLCEmbeddings(Embeddings):
    """LangChain adapter over the evaluation's embedding function.

    Caches by text, so tuning the semantic chunker's threshold embeds each
    sentence window once rather than once per attempt.
    """

    def __init__(self, embedding_fn) -> None:
        self._fn = embedding_fn
        self._cache: dict[str, list[float]] = {}

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        missing = [t for t in dict.fromkeys(texts) if t not in self._cache]
        if missing:
            self._cache.update(zip(missing, self._fn(missing)))
        return [self._cache[t] for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


# Paragraph first, then sentence ends, and only then line breaks and spaces.
# LangChain's default order puts "\n" and " " before any sentence end, so on
# text without short paragraphs it cuts inside sentences.
_SENTENCE_SEPARATORS = ["\n\n", r"(?<=[.!?])\s+", "\n", " ", ""]


def _avg_len(chunks: list[str]) -> int:
    return round(sum(map(len, chunks)) / max(1, len(chunks)))


def _fixed(text: str, size: int, overlap: int) -> list[str]:
    return CharacterTextSplitter(chunk_size=size, chunk_overlap=overlap, separator=" ").split_text(text)


def _recursive(text: str, size: int = 512, overlap: int = 50,
               sentence_aware: bool = False) -> list[str]:
    if sentence_aware:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=size, chunk_overlap=overlap, separators=_SENTENCE_SEPARATORS,
            is_separator_regex=True, keep_separator="end")
    else:
        splitter = RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=overlap)
    return splitter.split_text(text)


def _match_length(make, target: int, start: int) -> list[str]:
    """Tune `make(param)` until the mean chunk length is within 4 % of `target`."""
    param, best = start, None
    for _ in range(6):
        cand = make(param)
        got = _avg_len(cand)
        if best is None or abs(got - target) < abs(_avg_len(best) - target):
            best = cand
        if abs(got - target) / target <= 0.04:
            break
        param = max(60, round(param * target / max(1, got)))
    return best


def _cap_chars(chunks: list[str], max_chars: int, splitter: TextSplitter) -> list[str]:
    """Cut chunks longer than `max_chars` at sentence boundaries, as LLMChunker does."""
    out: list[str] = []
    for chunk in chunks:
        if len(chunk) <= max_chars:
            out.append(chunk)
            continue
        current = ""
        for sentence in splitter.split_sentences(chunk):
            if current and len(current) + 1 + len(sentence) > max_chars:
                out.append(current)
                current = sentence
            else:
                current = f"{current} {sentence}".strip()
        if current:
            out.append(current)
    return out


def build_strategies(text: str, match_len: int | None = None) -> dict[str, list[str]]:
    out = {
        "fixed_256": _fixed(text, 256, 20),
        "fixed_512": _fixed(text, 512, 50),
        "recursive": _recursive(text),
    }
    if match_len: #matches length of llm-semantic-chunker to show difference in chunkking not length
        out[f"fixed_matched_{match_len}"] = _fixed(text, match_len, 50)
        out[f"recursive_matched_{match_len}"] = _match_length(
            lambda size: _recursive(text, size, 50), match_len, start=match_len)
    return out


def build_recursive_sentences(text: str, match_len: int) -> list[str]:
    """Recursive splitting that prefers sentence ends to spaces, length-matched."""
    return _match_length(lambda size: _recursive(text, size, 50, sentence_aware=True),
                         match_len, start=match_len)


class _AlwaysSameTopic:
    """LLMClient that never opens a topic boundary, so only the size cap cuts."""

    def chat(self, messages: list[dict]) -> str:
        return "YES"


def build_packing(text: str, match_len: int) -> list[str]:
    """The incremental chunker with the model taken out, length-matched.

    Same sentences, step and midpoint split as the LLM arm; every boundary
    comes from the character cap, which is tuned to the target mean length.
    Whatever the LLM arm gains over this arm is what the model's topic
    decisions contribute.
    """
    def pack(max_chars: int) -> list[str]:
        return LLMChunker(client=_AlwaysSameTopic(), respect_headings=False,
                          step_sentences=2, max_chunk_sentences=100,
                          max_chunk_chars=max_chars, smart_split=False,
                          filter_low_info=False).chunk(text)
    return _match_length(pack, match_len, start=2 * match_len)


def build_semantic_lc(text: str, embedding_fn, match_len: int,
                      max_chars: int | None = None) -> list[str]:
    """LangChain's SemanticChunker, held to the same terms as the LLM arm.

    The breakpoint percentile is searched until the mean chunk length matches
    `match_len`, and chunks above `max_chars` (the LLM arm's cap) are cut at
    sentence boundaries. At its default of percentile 95 with no cap, the
    chunks run to thousands of characters and the embedder truncates them, so
    the baseline loses on length rather than on where it draws boundaries.
    """
    embeddings = _CachedLCEmbeddings(embedding_fn)
    sentences = TextSplitter()

    def split(percentile: float) -> list[str]:
        chunks = SemanticChunker(
            embeddings,
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_amount=percentile,
        ).split_text(text) or [text]
        return _cap_chars(chunks, max_chars, sentences) if max_chars else chunks

    # a higher percentile means fewer breakpoints and longer chunks
    lo, hi, best = 0.0, 100.0, None
    for _ in range(12):
        percentile = (lo + hi) / 2
        cand = split(percentile)
        got = _avg_len(cand)
        if best is None or abs(got - match_len) < abs(_avg_len(best) - match_len):
            best = cand
        if abs(got - match_len) / match_len <= 0.04:
            break
        if got < match_len:
            lo = percentile
        else:
            hi = percentile
    return best

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
        # Text actually searched, per arm. Length-matched baselines equalise the
        # MEAN chunk length, not the total mass: on nasa the LLM arm searches
        # about 17 % fewer characters than recursive_matched, because the
        # low-information filter and whitespace normalisation run only there.
        # Less text means fewer distractors — without this column the boundary
        # quality effect cannot be separated from that.
        unique_texts = set(display_texts) if display_texts else chunks
        corpus_chars = sum(len(c) for c in unique_texts)
        n_units = max(1, len(set(display_texts)) if display_texts else len(chunks))

        return {
            "strategy":            name,
            "chunk_count":         len(chunks),
            "corpus_chars":        corpus_chars,
            "avg_chunk_len":       round(corpus_chars / n_units),
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
