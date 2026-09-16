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

    _chunker: LLMChunker = PrivateAttr()

    def __init__(self, client=None, **chunker_kwargs) -> None:
        super().__init__()
        chunker_kwargs.setdefault("mode", "incremental")
        self._chunker = LLMChunker(client=client or OllamaClient(),
                                   config=ChunkerConfig(**chunker_kwargs))

    def split_text(self, text: str) -> list[str]:
        return self._chunker.chunk(text)
