from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Protocol, runtime_checkable


@runtime_checkable
class LLMClient(Protocol):
    """Structural protocol — any object with chat(messages) qualifies."""
    def chat(self, messages: list[dict]) -> str: ...


class BasePrompt(ABC):
    #all prompt-classes need as_messages and are only giving out lists
    @abstractmethod
    def as_messages(self, text: str) -> list: ... #empty method head


class ChunkPostProcessor(ABC):
   #every post-process(metadata, remove, merging)
    @abstractmethod
    def process(self, chunks: list[str]) -> list[str]: ...
