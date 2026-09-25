from __future__ import annotations

from langchain_text_splitters import TextSplitter

from ..chunker import LLMChunker
from ..config import ChunkerConfig
from ..llm_client import OllamaClient

"""LangChain adapter.
    pip install llm-semantic-chunker[langchain]
"""

__all__ = ["LLMSemanticSplitter"]


class LLMSemanticSplitter(TextSplitter):
    """LangChain adapter: this chunker wherever a TextSplitter is expected.

    Implements `split_text`, so `create_documents`, `split_documents` and
    `transform_documents` come from the base class. Only the splitting step
    changes; the rest of a LangChain pipeline is untouched.
    """

    def __init__(self, client=None, **chunker_kwargs) -> None:
        super().__init__()
        chunker_kwargs.setdefault("mode", "incremental")
        self._chunker = LLMChunker(client=client or OllamaClient(),
                                   config=ChunkerConfig(**chunker_kwargs))

    def split_text(self, text: str) -> list[str]:
        return self._chunker.chunk(text)
