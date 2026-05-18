"""
Chunking Strategy Evaluation

Compares chunking strategies on retrieval quality using auto-generated questions.

Usage:
    python evaluate.py <path/to/file.pdf>
    python evaluate.py <path/to/file.pdf> --regen-questions   # re-generate test questions
    python evaluate.py <path/to/file.pdf> --top-k 5           # evaluate Hit Rate@5 (default: 3)

Strategies compared:
    - fixed_256    : Fixed-size chunks, 256 chars, 20 char overlap  (LangChain)
    - fixed_512    : Fixed-size chunks, 512 chars, 50 char overlap  (LangChain)
    - recursive    : Recursive character splitting, 512 chars       (LangChain)
    - semantic     : Semantic chunking via embeddings               (LangChain)
    - llm          : Your LLM-based semantic chunker

Metrics:
    - Hit Rate@K   : % of questions where correct chunk is in Top-K results
    - MRR          : Mean Reciprocal Rank (1/rank of first correct hit)
    - Avg Dist     : Average cosine distance of Top-1 result (lower = better)
    - Chunk Count  : Number of chunks produced
    - Avg Chunk Len: Average chunk length in characters
"""

from __future__ import annotations

import json
import sys
import re
from pathlib import Path
from typing import Optional
from datetime import datetime

from pypdf import PdfReader
from langchain_text_splitters import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
)
from langchain_experimental.text_splitter import SemanticChunker
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings

from llm_chunker import LLMChunker, QwenClient
from llm_chunker.vectorstore import VectorStore

# ── Paths ──────────────────────────────────────────────────────────────────────
CACHE_DIR    = Path("./chunks_cache")
EVAL_DIR     = Path("./eval_cache")
RESULTS_DIR  = Path("./eval_results")


# ── PDF helpers ────────────────────────────────────────────────────────────────
def extract_text(pdf_path: str) -> str:
    reader = PdfReader(pdf_path)
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(p.strip() for p in pages if p.strip())


# ── Chunking strategies ────────────────────────────────────────────────────────
def chunk_fixed(text: str, size: int, overlap: int) -> list[str]:
    splitter = CharacterTextSplitter(
        chunk_size=size, chunk_overlap=overlap, separator=" "
    )
    return splitter.split_text(text)


def chunk_recursive(text: str, size: int = 512, overlap: int = 50) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=size, chunk_overlap=overlap
    )
    return splitter.split_text(text)


def chunk_semantic_langchain(text: str) -> list[str]:
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    splitter = SemanticChunker(embeddings)
    return splitter.split_text(text)


def load_llm_chunks(pdf_path: str) -> Optional[list[str]]:
    path = CACHE_DIR / (Path(pdf_path).stem + ".json")
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        print(f"[llm-chunks] Loaded {len(data['chunks'])} chunks from cache")
        return data["chunks"]
    return None


def load_llm_chunks_enriched(pdf_path: str) -> Optional[list[str]]:
    path = CACHE_DIR / (Path(pdf_path).stem + "_enriched.json")
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        print(f"[llm-enriched] Loaded {len(data['chunks'])} enriched chunks from cache")
        return data["chunks"]
    return None


STRATEGIES = {
    "fixed_256":  lambda text: chunk_fixed(text, 256, 20),
    "fixed_512":  lambda text: chunk_fixed(text, 512, 50),
    "recursive":  lambda text: chunk_recursive(text),
    "semantic_lc": lambda text: chunk_semantic_langchain(text),
    # llm strategy is handled separately (needs cache)
}


# ── Question generation ────────────────────────────────────────────────────────
def generate_questions(
    llm_chunks: list[str],
    client: QwenClient,
    pdf_stem: str,
    max_questions: int = 50,
) -> list[dict]:
    """
    Generate one question per LLM chunk (up to max_questions).
    Each entry: { "question": str, "source_text": str, "chunk_index": int }
    """
    qa_path = EVAL_DIR / f"{pdf_stem}_questions.json"
    if qa_path.exists():
        data = json.loads(qa_path.read_text(encoding="utf-8"))
        print(f"[questions] Loaded {len(data)} questions from cache")
        return data

    EVAL_DIR.mkdir(exist_ok=True)
    # Sample evenly if more chunks than max_questions
    step = max(1, len(llm_chunks) // max_questions)
    sampled = [(i, llm_chunks[i]) for i in range(0, len(llm_chunks), step)][:max_questions]

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
        question = client.chat(messages).strip()
        # Strip leading "Question:" prefix if model adds it
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


# ── Hit detection ──────────────────────────────────────────────────────────────
def _overlap_ratio(a: str, b: str) -> float:
    """Character-level overlap: how much of `a` appears in `b`."""
    if not a:
        return 0.0
    # Sliding window: check if any 60-char window of `a` is in `b`
    window = 60
    a_clean = " ".join(a.split())
    b_clean = " ".join(b.split())
    matches = sum(
        1 for i in range(0, len(a_clean) - window + 1, window)
        if a_clean[i:i + window] in b_clean
    )
    windows_total = max(1, len(a_clean) // window)
    return matches / windows_total


def is_hit(source_text: str, retrieved_text: str, threshold: float = 0.3) -> bool:
    """True if the retrieved chunk contains enough of the source text."""
    return _overlap_ratio(source_text, retrieved_text) >= threshold


# ── Evaluation core ────────────────────────────────────────────────────────────
def evaluate_strategy(
    name: str,
    chunks: list[str],
    qa_pairs: list[dict],
    store: VectorStore,
    k: int = 3,
) -> dict:
    collection = f"eval_{name}"
    store.add_chunks(collection, chunks, source=name)

    hits = 0
    reciprocal_ranks = []
    top1_distances = []

    for qa in qa_pairs:
        results = store.query(collection, qa["question"], k=k)
        top1_distances.append(results[0].distance if results else 1.0)

        rank = None
        for i, r in enumerate(results, 1):
            if is_hit(qa["source_text"], r.chunk_text):
                rank = i
                break

        if rank is not None:
            hits += 1
            reciprocal_ranks.append(1.0 / rank)
        else:
            reciprocal_ranks.append(0.0)

    n = len(qa_pairs)
    avg_len = sum(len(c) for c in chunks) / max(1, len(chunks))

    return {
        "strategy":      name,
        "chunk_count":   len(chunks),
        "avg_chunk_len": round(avg_len),
        f"hit_rate@{k}": round(hits / n * 100, 1),
        "mrr":           round(sum(reciprocal_ranks) / n, 3),
        "avg_dist_top1": round(sum(top1_distances) / n, 3),
    }


# ── Results output ─────────────────────────────────────────────────────────────
def print_table(results: list[dict], k: int) -> None:
    hr_key = f"hit_rate@{k}"
    header = f"{'Strategy':<15} {'Chunks':>7} {'Avg Len':>8} {f'Hit@{k}':>8} {'MRR':>7} {'Avg Dist':>9}"
    print("\n" + "=" * len(header))
    print(header)
    print("=" * len(header))
    for r in results:
        print(
            f"{r['strategy']:<15} "
            f"{r['chunk_count']:>7} "
            f"{r['avg_chunk_len']:>8} "
            f"{r[hr_key]:>7.1f}% "
            f"{r['mrr']:>7.3f} "
            f"{r['avg_dist_top1']:>9.3f}"
        )
    print("=" * len(header))


def save_charts(results: list[dict], k: int, pdf_stem: str) -> None:
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use("Agg")
    except ImportError:
        print("[charts] matplotlib not installed — skipping charts")
        return

    RESULTS_DIR.mkdir(exist_ok=True)
    hr_key  = f"hit_rate@{k}"
    names   = [r["strategy"] for r in results]
    colors  = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"]

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle(f"Chunking Strategy Comparison — {pdf_stem}", fontsize=13)

    def bar(ax, values, title, ylabel, fmt="{:.1f}"):
        bars = ax.bar(names, values, color=colors[:len(names)])
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, max(values) * 1.2 if max(values) > 0 else 1)
        for bar_, val in zip(bars, values):
            ax.text(
                bar_.get_x() + bar_.get_width() / 2,
                bar_.get_height() + max(values) * 0.02,
                fmt.format(val), ha="center", va="bottom", fontsize=9
            )
        ax.tick_params(axis="x", rotation=20)

    bar(axes[0], [r[hr_key] for r in results],    f"Hit Rate@{k} (%)",        "%", "{:.1f}")
    bar(axes[1], [r["mrr"] for r in results],      "MRR (higher = better)",   "",  "{:.3f}")
    bar(axes[2], [r["avg_dist_top1"] for r in results], "Avg Distance Top-1\n(lower = better)", "", "{:.3f}")

    plt.tight_layout()
    out = RESULTS_DIR / f"{pdf_stem}_comparison.png"
    plt.savefig(out, dpi=150)
    print(f"\n[charts] Saved to {out}")


def save_json(results: list[dict], pdf_stem: str) -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    out  = RESULTS_DIR / f"{pdf_stem}_{ts}.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[results] Saved to {out}")


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python evaluate.py <path/to/file.pdf> [--regen-questions] [--top-k N]")
        sys.exit(1)

    pdf_path       = sys.argv[1]
    regen_q        = "--regen-questions" in sys.argv
    top_k          = int(sys.argv[sys.argv.index("--top-k") + 1]) if "--top-k" in sys.argv else 3

    if not Path(pdf_path).exists():
        print(f"File not found: {pdf_path}")
        sys.exit(1)

    pdf_stem = Path(pdf_path).stem

    # 1. Extract text
    print(f"Reading: {pdf_path}")
    text = extract_text(pdf_path)
    print(f"Extracted {len(text)} chars\n")

    # 2. Load LLM chunks from cache (required — run pdf_example.py first)
    llm_chunks = load_llm_chunks(pdf_path)
    if llm_chunks is None:
        print("No LLM chunk cache found. Run pdf_example.py first to generate chunks.")
        sys.exit(1)

    # 3. Generate / load test questions
    client = QwenClient()
    if regen_q:
        qa_path = EVAL_DIR / f"{pdf_stem}_questions.json"
        if qa_path.exists():
            qa_path.unlink()
    qa_pairs = generate_questions(llm_chunks, client, pdf_stem, max_questions=50)

    # 4. Build all chunk sets
    print("\n[chunking] Building all strategies...")
    all_chunks = {name: fn(text) for name, fn in STRATEGIES.items()}
    all_chunks["llm"] = llm_chunks

    llm_enriched = load_llm_chunks_enriched(pdf_path)
    if llm_enriched is not None:
        all_chunks["llm_enriched"] = llm_enriched
    else:
        print("  [llm_enriched] No enriched cache found — skipping. Run pdf_example.py with enrich=True first.")

    for name, chunks in all_chunks.items():
        print(f"  {name:<15}: {len(chunks)} chunks")

    # 5. Evaluate each strategy
    store = VectorStore(persist_dir="./chroma_db_eval")
    print(f"\n[eval] Running retrieval tests (k={top_k}) over {len(qa_pairs)} questions...\n")

    results = []
    for name, chunks in all_chunks.items():
        print(f"  Evaluating: {name}...")
        result = evaluate_strategy(name, chunks, qa_pairs, store, k=top_k)
        results.append(result)

    # 6. Output
    print_table(results, top_k)
    save_json(results, pdf_stem)
    save_charts(results, top_k, pdf_stem)
