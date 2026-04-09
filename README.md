# semantic-chunking

A Python library for **LLM-based semantic chunking** designed for Retrieval-Augmented Generation (RAG) pipelines. Instead of splitting text at fixed character counts, this library uses a local LLM to understand where topics change and groups sentences accordingly.

---

## Installation

```bash
pip install pypdf sentence-transformers httpx
```

Requires [Ollama](https://ollama.com) running locally with a compatible model:

```bash
ollama pull qwen3.5:4b   # or: llama3.2:3b, qwen2.5:3b, gemma3:4b
```

---

## Quick Start

```python
from llm_chunker import LLMChunker, QwenClient

chunker = LLMChunker(client=QwenClient())
chunks = chunker.chunk("Your text here...")

for i, chunk in enumerate(chunks, 1):
    print(f"--- Chunk {i} ---")
    print(chunk)
```

### Chunking a PDF

```bash
python pdf_example.py /path/to/document.pdf
```

---

## Architecture

The library processes text through a **5-step pipeline**:

```
Input Text
    │
    ▼
[1] Pre-split into mini-chunks        (regex, no LLM)
    │
    ▼
[2] Sliding window boundary detection (LLM)
    │
    ▼
[3] Low-info filter                   (LLM)
    │
    ▼
[4] Short-chunk merge                 (rule-based)
    │
    ▼
[5] Similar-chunk merge               (LLM)
    │
    ▼
Output Chunks
```

---

## Module Reference

### `llm_chunker/llm_client.py` — `QwenClient`

HTTP client that communicates with a locally running Ollama model via `/api/chat`.

```python
QwenClient(
    model="qwen3.5:4b",     # Ollama model tag
    base_url="http://localhost:11434",
    temperature=0.0,        # 0.0 = deterministic output
    timeout=600.0,          # read timeout in seconds (CPU inference is slow)
    thinking=False,         # disable chain-of-thought (<think> blocks)
    num_ctx=4096,           # context window size — reduce to save RAM
)
```

**Key method:**

| Method | Description |
|--------|-------------|
| `chat(messages)` | Sends a list of `{"role": ..., "content": ...}` messages and returns the model's response as a string |

`_strip_thinking()` removes `<think>...</think>` blocks that some models (e.g. Qwen3) emit before their actual answer.

---

### `llm_chunker/prompts.py`

Contains three prompt dataclasses. Each builds the `messages` list passed to `QwenClient.chat()`.

#### `BoundaryPrompt`

Used in **Step 2** (sliding window boundary detection).

The LLM receives a window of numbered mini-chunks wrapped in XML tags:

```
<chunk_0>First sentences...</chunk_0>
<chunk_1>Next sentences...</chunk_1>
...
```

It returns the indices where a new topic starts, e.g. `3, 7`. The chunker uses these indices to split the mini-chunks into final chunks.

#### `LowInfoPrompt`

Used in **Step 3** (low-info filter).

Each chunk is evaluated individually. The LLM answers `YES` (keep) or `NO` (remove). Chunks are removed if they are:
- Titles or headings with no content
- Table of contents entries
- Page numbers, headers, footers
- Boilerplate text

#### `MergePrompt`

Used in **Step 5** (similar-chunk merge).

Two adjacent chunks are shown to the LLM. It answers `YES` (merge) or `NO` (keep separate). Merging happens when both chunks discuss the same subject or one continues the other.

---

### `llm_chunker/chunker.py` — `LLMChunker`

Main class. Orchestrates the full pipeline.

```python
LLMChunker(
    client=QwenClient(),            # LLM client (default: QwenClient)
    filter_low_info=True,           # enable/disable Step 3
    merge_similar=True,             # enable/disable Step 5
    min_chunk_chars=100,            # minimum chunk length before forced merge (Step 4)
    window_size=10,                 # number of mini-chunks per LLM window (Step 2)
    step_size=5,                    # how far to advance the window each iteration
    sentences_per_mini_chunk=3,     # sentences grouped per mini-chunk (Step 1)
    verbose=False,                  # print internal pipeline steps
)
```

#### Methods

| Method | Step | Description |
|--------|------|-------------|
| `chunk(text)` | — | Runs the full pipeline. Returns `list[str]` |
| `_find_boundaries(mini_chunks)` | 2 | Slides window over mini-chunks, collects boundary indices from LLM |
| `_tag_window(window, offset)` | 2 | Wraps mini-chunks in `<chunk_N>` XML tags for the LLM |
| `_parse_boundaries(raw, offset, max_idx)` | 2 | Extracts valid integer indices from the LLM response |
| `_merge_at_boundaries(mini_chunks, boundaries)` | 2 | Joins mini-chunks between boundary points into final chunks |
| `_remove_low_info(chunks)` | 3 | Calls LLM for each chunk, removes those answered NO |
| `_merge_short_chunks(chunks)` | 4 | Absorbs chunks shorter than `min_chunk_chars` into their neighbor |
| `_merge_similar_chunks(chunks)` | 5 | Iterates adjacent pairs, merges those the LLM says share a topic |

#### Helper functions (module-level)

| Function | Description |
|----------|-------------|
| `_paragraph_split(text)` | Splits raw text on blank lines, section markers, page markers |
| `_sentence_split(text)` | Splits a paragraph into sentences; protects abbreviations (Dr., Abb., etc.) |
| `_make_mini_chunks(text, sentences_per_chunk)` | Combines both: paragraphs → sentences → fixed-size sentence groups |

---

## Configuration Guide

### Reducing RAM usage (8 GB machines)

```python
QwenClient(
    model="llama3.2:3b",   # smaller model
    num_ctx=1024,           # smaller context window
)
```

### Faster processing (disable LLM steps)

```python
LLMChunker(
    filter_low_info=False,  # skip Step 3
    merge_similar=False,    # skip Step 5
)
```

### More aggressive merging

```python
LLMChunker(
    min_chunk_chars=300,    # merge more short chunks
    merge_similar=True,
)
```

### Debug mode

```python
LLMChunker(verbose=True)
# Prints each pipeline step, window responses, merge decisions
```

---

## Project Structure

```
semantic-chunking/
├── llm_chunker/
│   ├── __init__.py        # public exports
│   ├── chunker.py         # LLMChunker — main pipeline
│   ├── llm_client.py      # QwenClient — Ollama HTTP client
│   └── prompts.py         # BoundaryPrompt, LowInfoPrompt, MergePrompt
├── pdf_example.py         # example: chunk a PDF file
└── README.md
```

---

## Comparison with Existing Methods

| Method | Boundary detection | Semantic coherence | RAM |
|--------|-------------------|--------------------|-----|
| Fixed-size (LlamaIndex) | Character count | None | Minimal |
| Recursive (LangChain) | Separators (`.`, `\n`) | Low | Minimal |
| Embedding-based (semantic) | Cosine similarity drop | Medium | ~500 MB |
| **This library** | LLM (sliding window) | High | ~2–5 GB |

The evaluation in the thesis compares these methods on retrieval quality using a vector database (ChromaDB).
