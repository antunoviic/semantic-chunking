# semantic-chunking

A Python library for **LLM-based semantic chunking**, designed for Retrieval-Augmented Generation (RAG) pipelines. Instead of splitting text at a fixed character count, it uses a local LLM to decide where a topic actually changes and groups sentences accordingly.

The library runs entirely against a **local Ollama model** — no API keys, no data leaving your machine.

---

## Installation

```bash
pip install httpx nltk langdetect chromadb sentence-transformers pypdf
```

Requires [Ollama](https://ollama.com) running locally with a compatible model:

```bash
ollama pull qwen3.5:4b  
```

On an 8 GB machine, prefer a smaller model (`llama3.2:3b`) and a reduced context window (see [Configuration](#configuration) below).

`sentence-transformers` pulls in `torch`, and `chromadb` pulls in `grpcio` — both are substantial, and on some platform/Python combinations `pip` builds `grpcio` from source rather than using a prebuilt wheel, which can take several minutes on top of the download, but only happens at first install.

---

## Quick Start

```python
from llm_chunker import LLMChunker, QwenClient

chunker = LLMChunker(client=QwenClient(), mode="incremental")
chunks = chunker.chunk("Your text here...")

for i, chunk in enumerate(chunks, 1):
    print(f"--- Chunk {i} ---")
    print(chunk)
```

### Chunking a PDF or text file

```python
from app.document_reader import DocumentReader
from llm_chunker import LLMChunker, QwenClient

text = DocumentReader("document.pdf").extract_text()
chunks = LLMChunker(client=QwenClient(), mode="incremental").chunk(text)
```

`DocumentReader` handles both `.pdf` and `.txt`.

---

## How it works

The library ships two chunking strategies, selected via `mode` and different types of ablations like metadata and size caps:

### `mode="incremental"` (standard)

The LLM reads the document sentence by sentence and decides, for each new group of sentences, whether it still belongs to the chunk being built or starts a new one:


Three things shape the boundaries on top of that:

- **Heading awareness** — a line-based regex detects section headings in the raw text (before sentence splitting, so numbered headings like "3.2. Error Handling" are still recognized) and forces a hard boundary there. With `heading_mode="hybrid"`, sentences that look like a heading but weren't caught by the regex are additionally checked by the LLM — useful for documents whose headings aren't reliably formatted.
- **Size cap** — `max_chunk_sentences` / `max_chunk_chars` force a split once a chunk grows past a limit, regardless of topic continuity, so a single "still on topic" run can't produce an unusably large chunk.
- **Low-info filter** — a post-processing pass removes chunks that turned out to be near-empty (bare headings, page numbers, boilerplate) rather than actual content.

### `mode="window"` (legacy)

An earlier, two-pass approach: the text is pre-split into fixed-size mini-chunks, a sliding window over them proposes coarse boundaries, and a second pass then re-evaluates each pair of adjacent chunks and merges them back if they turn out to share a topic. Only kept for comparison.

---

## Configuration

```python
LLMChunker(
    client=QwenClient(),           # LLM backend
    mode="incremental",            # "incremental" (recommended) or "window" (legacy)

    # --- incremental mode ---
    step_sentences=3,              # sentences considered per boundary decision
    max_chunk_sentences=20,        # hard cap regardless of topic continuity
    max_chunk_chars=None,          # optional character cap, enforced at sentence boundaries
    respect_headings=True,         # force a boundary at detected section headings
    heading_mode="regex",          # "regex" or "hybrid" (regex + LLM fallback)
    smart_split=True,              # let the LLM choose where to split an over-long chunk
                                    # instead of cutting at the midpoint

    # --- window mode ---
    window_size=10,                # mini-chunks visible to the LLM per boundary decision
    step_size=5,                   # how far the window advances each iteration

    # --- shared ---
    filter_low_info=True,          # drop near-empty chunks after assembly
    enrich=False,                  # prefix each chunk with an LLM-generated topic line
    language=None,                 # sentence-splitter language; auto-detected if None
    verbose=False,                 # print each boundary decision as it's made
)
```

### `QwenClient`

```python
QwenClient(
    model="qwen3.5:4b",
    base_url="http://localhost:11434",   # or set OLLAMA_BASE_URL
    temperature=0.0,                     # near-deterministic decoding
    seed=42,                             # temperature=0 alone is not bit-exact on Ollama —
                                          # this closes the gap. Verified: 3 full chunking
                                          # runs of the same document produced identical output.
    timeout=600.0,                       # read timeout; local inference can be slow
    num_ctx=4096,                        # context window — lower to save RAM
)
```

Transient Ollama errors (5xx, timeouts) are retried automatically with exponential backoff.

If you only want to chunk text, `llm_chunker/` and `app/document_reader.py` are all you need — everything else supports evaluating and comparing chunking strategies, which is a separate concern from producing chunks.





