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

#same config
SETTINGS = dict(mode="incremental", max_chunk_chars=1200)


def show(title: str, chunks: list[str]) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")
    for i, chunk in enumerate(chunks, 1):
        print(f"[{i}] {chunk}\n")


# native

def run_native() -> None:
    chunker = LLMChunker(client=OllamaClient(), **SETTINGS)
    show("1. Native — llm_semantic_chunker", chunker.chunk(TEXT))


# LangChain

def run_langchain() -> None:
    try:
        from llm_semantic_chunker.integrations.langchain import LLMSemanticSplitter
    except ImportError:
        print("\n[skip] LangChain — pip install llm-semantic-chunker[langchain]")
        return #would give ImportError if not installed

    splitter = LLMSemanticSplitter(**SETTINGS)

    show("2. LangChain — split_text()", splitter.split_text(TEXT))

    docs = splitter.create_documents([TEXT])
    print(f"create_documents() -> {len(docs)} x {type(docs[0]).__name__}, "
          f"ready for any LangChain vector store")


# LlamaIndex

def run_llamaindex() -> None:
    try:
        from llama_index.core import Document
        from llm_semantic_chunker.integrations.llamaindex import LLMSemanticNodeParser
    except ImportError:
        print("\n[skip] LlamaIndex — pip install llm-semantic-chunker[llamaindex] "
              "(needs Python 3.10+)")
        return

    parser = LLMSemanticNodeParser(**SETTINGS)

    show("3. LlamaIndex — split_text()", parser.split_text(TEXT))

    nodes = parser.get_nodes_from_documents([Document(text=TEXT)])
    print(f"get_nodes_from_documents() -> {len(nodes)} x {type(nodes[0]).__name__}, "
          f"ready for any LlamaIndex index")


if __name__ == "__main__":
    run_native()
    run_langchain()
    run_llamaindex()
