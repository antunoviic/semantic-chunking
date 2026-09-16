# llm-semantic-chunker

A Python library for **LLM-based semantic chunking**, designed for Retrieval-Augmented Generation (RAG) pipelines. Instead of splitting text at a fixed character count, it uses a local LLM to decide where a topic actually changes and groups sentences accordingly.

The library runs entirely against a **local Ollama model** — no API keys, no data leaving your machine.

---

## Installation

```bash
pip install llm-semantic-chunker
```

That is the whole install — `httpx`, `nltk` and `langdetect`. No vector store, no `torch`. On first use the sentence splitter downloads NLTK's `punkt` data once.

Requires [Ollama](https://ollama.com) running locally with a compatible model, here:

```bash
ollama pull qwen3.5:4b
```

Optional extras, only if you want them:

```bash
pip install "llm-semantic-chunker[langchain]"    # LangChain adapter
pip install "llm-semantic-chunker[llamaindex]"   # LlamaIndex adapter (needs Python 3.10+)
pip install "llm-semantic-chunker[pdf]"          # read .pdf input
```

---

## Quick Start

```python
from llm_semantic_chunker import LLMChunker, OllamaClient

TEXT = (
    "The sun is a star at the center of the solar system. It is a nearly "
    "perfect sphere of hot plasma, heated to incandescence by nuclear fusion "
    "in its core. Its diameter is about 1.39 million kilometres, roughly 109 "
    "times that of Earth. "
    "Dolphins are highly intelligent marine mammals. They live in social "
    "groups called pods and use echolocation to navigate and hunt. Some "
    "species have been observed teaching their young to use tools."
)

chunker = LLMChunker(client=OllamaClient(), mode="incremental", max_chunk_chars=1200)

for i, chunk in enumerate(chunker.chunk(TEXT), 1):
    print(f"--- Chunk {i} ---")
    print(chunk)
```

The LLM splits between the two topics rather than at a character count:

```
--- Chunk 1 ---
The sun is a star at the center of the solar system. It is a nearly perfect
sphere of hot plasma, heated to incandescence by nuclear fusion in its core.
Its diameter is about 1.39 million kilometres, roughly 109 times that of Earth.
--- Chunk 2 ---
Dolphins are highly intelligent marine mammals. They live in social groups
called pods and use echolocation to navigate and hunt. Some species have been
observed teaching their young to use tools.
```

---

## Use it inside LangChain

`pip install "llm-semantic-chunker[langchain]"`

The adapter implements LangChain's `TextSplitter`, so it goes wherever `RecursiveCharacterTextSplitter` goes — only the splitting step changes, the rest of the pipeline is untouched.

```python
from langchain_core.documents import Document
from llm_semantic_chunker.integrations.langchain import LLMSemanticSplitter

TEXT = ("The sun is a star at the center of the solar system. It is a nearly "
        "perfect sphere of hot plasma. Dolphins are highly intelligent marine "
        "mammals. They live in social groups called pods.")


splitter = LLMSemanticSplitter(max_chunk_chars=1200)

docs: list[Document] = splitter.create_documents([TEXT])
for d in docs:
    print(d.page_content)
```

`split_text()`, `split_documents()` and `transform_documents()` work as well.

---

## Use it inside LlamaIndex

`pip install "llm-semantic-chunker[llamaindex]"` — **requires Python 3.10 or newer** ; the rest of this library still runs on Python 3.9.

```python
from llama_index.core import Document
from llm_semantic_chunker.integrations.llamaindex import LLMSemanticNodeParser

TEXT = ("The sun is a star at the center of the solar system. It is a nearly "
        "perfect sphere of hot plasma. Dolphins are highly intelligent marine "
        "mammals. They live in social groups called pods.")


parser = LLMSemanticNodeParser(max_chunk_chars=1200)

nodes = parser.get_nodes_from_documents([Document(text=TEXT)])
for n in nodes:
    print(n.text)
```

All three routes return the same chunks — only the object type differs.

---

## How it works

The library ships two chunking strategies, selected via `mode`.

### `mode="incremental"` (standard)

The LLM reads the document sentence by sentence and decides, for each new group of sentences, whether it still belongs to the chunk being built or starts a new one.

Four settings shape the result on top of that:

- **Heading awareness** — `heading_mode` picks how section headings are found, and a detected heading forces a boundary. In `"regex"` mode the boundary sits exactly at the heading sentence; in `"lines"` and `"hybrid"` mode it sits in front of the sentence group (`step_sentences`) that contains the heading, so with `step_sentences=2` the group's first sentence may precede the heading:
  - `"regex"` (default) — a sentence-level pattern, applied after sentence splitting
  - `"lines"` — a stronger line-based pattern applied to the raw text *before* sentence splitting, so numbered headings like "3.2. Error Handling" survive tokenisation
  - `"hybrid"` — the line-based pattern, plus an LLM check for short sentence groups (at most 90 characters and 12 words) the pattern did not flag. A heading that shares its group with a full sentence is not checked, so on well-formatted documents the LLM adds little over `"lines"`
- **Size cap** — `max_chunk_sentences` / `max_chunk_chars` force a split once a chunk outgrows the limit, even if the topic continues. The cut never falls inside a sentence: the LLM picks the best sentence boundary, and the chunker walks it back until the piece fits. A single sentence longer than the cap therefore stays whole — the cap is a target, not a guarantee.
- **Low-info filter** — a post-processing pass removes chunks that turned out to be near-empty boilerplate rather than actual content.
- **Topic enrichment** — `enrich=True` prefixes every chunk with an LLM-generated `[Topic: ...]` line, so the embedding also carries where the chunk sits in the document. Off by default: it costs one extra LLM call per chunk.

### `mode="window"` (legacy)

An earlier, two-pass approach: the text is pre-split into fixed-size mini-chunks, a sliding window over them proposes coarse boundaries. Only kept for comparison — without a size cap it degenerates into a few very large chunks.

---

## Configuration

Settings live in `ChunkerConfig`. Pass one explicitly, or give the individual
settings to `LLMChunker` and one is built for you — both are equivalent:

```python
from llm_semantic_chunker import ChunkerConfig, LLMChunker, OllamaClient

# short form
LLMChunker(client=OllamaClient(), max_chunk_chars=1200)

# explicit — useful when you want to reuse, compare or log the settings
config = ChunkerConfig(max_chunk_chars=1200)
chunker = LLMChunker(client=OllamaClient(), config=config)
chunker.config.max_chunk_chars      # 1200
```

`ChunkerConfig` is frozen and validates itself, so a bad value fails before the run. Passing a config *and* individual
settings at the same time is refused, because it would be ambiguous which one wins.

```python
ChunkerConfig(
    mode="incremental",            # "incremental" (recommended) or "window" 

    # --- incremental mode ---
    step_sentences=3,              # sentences considered per boundary decision
    max_chunk_sentences=20,        # hard cap regardless of topic continuity
    max_chunk_chars=None,          # character cap, applied at sentence boundaries;
                                  
    respect_headings=True,         # force a boundary at detected section headings
    heading_mode="regex",          # "regex", "lines" or "hybrid"

    smart_split=True,              # let the LLM choose where to split an 
                                    # instead of cutting at the midpoint

    # --- window mode ---
    window_size=10,                # mini-chunks visible to the LLM per boundary decision
    step_size=5,                   # how far the window advances each iteration

    # --- shared ---
    filter_low_info=True,          # drop low-info chunks after assembly
    enrich=False,                  # prefix each chunk with an LLM-generated topic line
    language=None,                 # sentence-splitter language; auto-detected if None
    verbose=False,                 # log every boundary decision to the console;
                                
                                    
)
```

### `OllamaClient`

```python
OllamaClient(
    model="qwen3.5:4b",
    base_url="http://localhost:11434",   # or set OLLAMA_BASE_URL
    temperature=0.0,                     # near-deterministic decoding
    seed=42,                             # temperature=0 
                                        # and seed to be bit accurate
                                          
    timeout=600.0,                       # read timeout;
    num_ctx=4096,                        # context window; lower saves RAM, but can
)                                        #be expanded
```


---


## License

MIT — see [LICENSE](https://github.com/antunoviic/semantic-chunking/blob/main/LICENSE).
