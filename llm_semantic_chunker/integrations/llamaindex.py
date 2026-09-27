from __future__ import annotations

from llama_index.core.node_parser import TextSplitter
from pydantic import PrivateAttr

from ..chunker import LLMChunker
from ..config import ChunkerConfig
from ..llm_client import OllamaClient

"""LlamaIndex adapter.
    pip install llm-semantic-chunker[llamaindex]

Needs Python 3.10 or newer
"""

__all__ = ["LLMSemanticNodeParser"]


class LLMSemanticNodeParser(TextSplitter):
    """LlamaIndex adapter: this chunker as a NodeParser.

    LlamaIndex's TextSplitter derives from NodeParser, so implementing
    `split_text` is enough for `get_nodes_from_documents` to work. Requires
    Python 3.10 or newer, because `llama-index-core` pulls in a dependency that
    uses PEP 604 syntax; the library itself still runs on 3.9.
    """

    _chunker: LLMChunker = PrivateAttr()

    def __init__(self, client=None, **settings) -> None:  #puts all arguments into a dict
        super().__init__()
        settings.setdefault("mode", "incremental")
        self._chunker = LLMChunker(client=client or OllamaClient(),
                                   config=ChunkerConfig(**settings))

    def split_text(self, text: str) -> list[str]:
        return self._chunker.chunk(text)
