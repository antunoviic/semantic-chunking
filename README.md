# llm-semantic-chunker

A Python library for **LLM-based semantic chunking**, designed for Retrieval-Augmented Generation (RAG) pipelines. Instead of splitting text at a fixed character count, it uses a local LLM to decide where a topic actually changes and groups sentences accordingly.

The library runs entirely against a **local Ollama model** — no API keys, no data leaving your machine.

**Links:** [PyPI](https://pypi.org/project/llm-semantic-chunker/) · [Source on GitHub](https://github.com/antunoviic/semantic-chunking) · [Issues](https://github.com/antunoviic/semantic-chunking/issues)

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
                                   # None = no cap
    respect_headings=True,         # force a boundary at detected section headings
    heading_mode="regex",          # "regex", "lines" or "hybrid"

    smart_split=True,              # at the size cap, let the LLM pick the split
                                   # point instead of cutting in the middle

    # --- window mode ---
    window_size=10,                # mini-chunks visible to the LLM per boundary decision
    step_size=5,                   # how far the window advances each iteration

    # --- shared ---
    filter_low_info=True,          # drop low-info chunks after assembly
    enrich=False,                  # prefix each chunk with an LLM-generated topic line
    language=None,                 # sentence-splitter language; auto-detected if None
    verbose=False,                 # attach a DEBUG console handler to the
                                   # package logger
)
```

### `OllamaClient`

```python
OllamaClient(
    model="qwen3.5:4b",
    base_url="http://localhost:11434",   # or set OLLAMA_BASE_URL
    temperature=0.0,                     # near-deterministic decoding
    seed=42,                             # temperature=0 alone is not bit-exact on
                                         # Ollama; the fixed seed made three full
                                         # runs identical (measured)
    timeout=600.0,                       # read timeout in seconds
    num_ctx=4096,                        # context window; lower saves RAM
)
```


---


## Evaluation harness

The repository also contains the evaluation part that produced the results of the bachelor thesis this library was written for. It chunks a document with every ablation strategy, embeds the chunks, runs a set of questions against every strategy and reports how often the answer was retrieved.

```bash
git clone https://github.com/antunoviic/semantic-chunking
cd semantic-chunking
pip install -e ".[eval]"
```

The `[eval]` extra adds ChromaDB, the LangChain baseline splitters, `pypdf` and
matplotlib. A second Ollama model is needed for the embeddings:

```bash
ollama pull qwen3.5:4b     # boundary decisions
ollama pull bge-m3         # embeddings
```

### A runnable example

A short, freely redistributable document and a verified question set are included, to run after cloning. The document is
RFC 8259, the JSON specification, a technical prose text with many sections.

```bash
python -m app.main demo/rfc8259_json.txt \
       --max-chunk-chars 1200 --max-chunk-sentences 100 --step-sentences 2 \
       --no-headings
```

The run should just take about 20 minutes, roughly one model call per two sentences for the boundaries. After that one per chunk for the low-information filter, then embedding and retrieval. 
It writes a Markdown report, a JSON file and a chart to
`eval_results/`, and caches the chunks — a second run skips the chunking
entirely and finishes the retrieval in under two minutes.

```
# Chunking Strategy Comparison — rfc8259_json

| Strategy                    | Chunks | Avg Len | Hit@1 | Hit@3 |   MRR | Ctx/Query |
|-----------------------------|-------:|--------:|------:|------:|------:|----------:|
| llm_incremental_parentchild |     97 |     737 | 71.4% | 85.7% | 0.815 |      2477 |
| llm_incremental             |     25 |     737 | 71.4% | 71.4% | 0.759 |      2487 |
| recursive                   |     73 |     356 | 71.4% | 71.4% | 0.733 |      1239 |
| semantic_lc                 |     38 |     668 | 50.0% | 92.9% | 0.713 |     15290 |
| recursive_matched_737       |     36 |     726 | 42.9% | 64.3% | 0.562 |      2336 |
| fixed_256                   |    100 |     249 | 35.7% | 35.7% | 0.373 |       753 |
```

**These numbers are a smoke test, not a result.** The demo set has fourteen questions, so a single question moves the ranking heavily and is not conclusive for the overall chunking. Its purpose is to show that the harness runs end to end and produces the comparison. (The thesis used question sets of 292, 296 and 400 questions.)

### What it compares

| Strategy | What it is |
|---|---|
| `fixed_256`, `fixed_512` | fixed-size splitting with overlap |
| `recursive` | LangChain's `RecursiveCharacterTextSplitter` |
| `fixed_matched_N`, `recursive_matched_N` | the same, with `N` tuned to the mean LLM chunk length — for a fair comparison |
| `semantic_lc` | LangChain's embedding-based `SemanticChunker` |
| `llm_incremental` | this library |
| `*_parentchild` | parent-child retrieval, applied to the LLM arm **and** to the matched baselines |

Reported per strategy: Hit@1, Hit@3, Hit@10, MRR, the number of chunks, the mean chunk length, the total corpus searched, and the characters returned per query at *k* = 3. The last two are reported because retrieval quality can be bought with context: a strategy that returns larger chunks raises its hit rate simply by including more text, and pays for it in the generator's context window.

### Chunking only

Dropping the retrieval step gets rid of a question set and is the fastest way to see what the chunker does to a document:

```bash
python -m app.main <document> --chunk-only --max-chunk-chars 1200
```

The log then reports where the boundaries came from, a topic decision made by the LLM, the size cap, or a heading.

### The documents evaluated in the thesis

All three are in `docs/`, with their verified question sets in `eval_cache/`:
the NASA Systems Engineering Handbook (implicit structure), RFC 9110 (explicit
structure) and H. G. Wells' *A Short History of the World* (prose). The run script `thesis/scripts/run_v4.sh` runs four ablation arms on each of the three, plus `--midpoint-split` on the NASA handbook alone.

### Using your own document

Any `.txt` or `.pdf` works. Its question set is read from
`eval_cache/<stem>_questions.json`, or from `--questions-file`, and is a list of
objects whose `source_text` is a **verbatim** substring of the document:

```json
[{"question": "What is the registered media type for JSON text?",
  "source_text": "The media type for JSON text is application/json. Type name: application Subtype name: json"}]
```

A chunk counts as a hit when the longest common substring of chunk and anchor covers at least 80 % of the anchor. `tools/make_question_prompts.py` produces the predefined prompt files for an external model to generate, `tools/verify_questions.py` checks the
replies and drops anchors that are not literally present, duplicated, or ambiguous.

### Ablation arms

Each flag changes exactly one thing and writes its own cache, so arms stay
comparable:

| Flag | Isolates |
|---|---|
| `--no-headings` | the reference arm |
| `--line-headings` / `--llm-headings` | what heading detection contributes |
| `--no-filter` | whether the gain comes from boundaries or from a smaller corpus |
| `--midpoint-split` | whether letting the model choose the split point helps |
| `--enrich` | whether a `[Topic: ...]` prefix helps |

`thesis/scripts/run_v4.sh` runs these on all three documents, except `--midpoint-split`, which is measured on the NASA handbook only.

### Caching

Every arm is cached, so an interrupted evaluation resumes where it stopped and a finished arm is skipped on the next run. Chunks are only reused when they were produced by the same code, so it ensures consistency in the evaluation process. Each cache file carries a fingerprint of the boundary-drawing modules, and one that no longer matches is treated as absent rather than loaded into a comparison.


---

## License

MIT — see [LICENSE](https://github.com/antunoviic/semantic-chunking/blob/main/LICENSE).
