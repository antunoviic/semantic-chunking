from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Protocol, runtime_checkable


@runtime_checkable
class LLMClient(Protocol):
    """Structural protocol — any object with chat(messages) qualifies."""
    def chat(self, messages: list[dict]) -> str: ...



# There is no single prompt abstraction, because the prompts ask three
# different kinds of question and therefore take three different inputs. One
# base class covering all of them could only promise "returns a list", which
# says nothing and would falsely suggest the prompts are interchangeable. The
# three protocols below each state a contract that is true, and a component
# declares which kind of question it needs. Like LLMClient, they are
# structural: conforming means having the method, not inheriting from it.


# Not runtime_checkable: isinstance() against a Protocol only checks that the
# method exists, not that its signature matches, so it would report every
# prompt as every role. They are for readers and for static checking, where
# the signatures are compared. The heading question is the one prompt the
# caller also asks to interpret the answer, hence parse().
class ChunkQuestion(Protocol):
    """Asked about one finished chunk: is it empty, what is its topic?"""

    def as_messages(self, text: str) -> list: ...


class BoundaryQuestion(Protocol):
    #asked while a chunk is assembled, does the candidate continue it?

    def as_messages(self, current_chunk: str, candidate: str) -> list: ...


class SplitPointQuestion(Protocol):
    #asked when a chunk outgrew the size cap, where should it be cut?

    def as_messages(self, sentences: list) -> list: ...


class HeadingQuestion(Protocol):
    #asked about one short line, is it a section heading? parses its own answer

    def as_messages(self, text: str) -> list: ...

    def parse(self, raw: str) -> bool: ...


class ChunkPostProcessor(ABC):
    #pass over the finished chunk list, removing, merging or annotating

   # processors run after the boundaries are drawn, and return a
    # new list. Adding one requires no change to the chunker

    @abstractmethod
    def process(self, chunks: list[str]) -> list[str]: ...
