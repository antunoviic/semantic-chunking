from __future__ import annotations

from dataclasses import dataclass, field

from .interfaces import BasePrompt

# Strategy-specific prompts live with their detector:
#   window/prompt.py       -> BoundaryPrompt
#   incremental/prompt.py  -> IncrementalBoundaryPrompt
# The prompts below are shared by the post-processing steps (post_processors.py).


@dataclass
class EnrichmentPrompt(BasePrompt):
    """
    Prompt to label a chunk with its structural position in the document
    (chapter/section heading, hierarchical if identifiable).
    The [Topic: ...] prefix is prepended to the chunk before embedding,
    so the vector better represents the chunk's place in the document.
    """

    system_message: str = field(default=(
        "You are a text annotation tool for RAG pipelines. "
        "Given a text chunk, you output the chapter or section heading it belongs to. "
        "Be concise and factual. Never add information that is not in the text."
    ))

    instruction_template: str = field(default=(
        "Analyze this text chunk. Identify which chapter or section it belongs to.\n"
        "Use the document's own heading if visible in the text (e.g. \"3. Stoic Virtues\").\n"
        "If a sub-section is identifiable, use the format: \"Parent Chapter > Sub-section\".\n"
        "If no heading is visible, create a concise 3-6 word label that describes the topic.\n"
        "Output only:\n"
        "Topic: <heading or label>\n\n"
        "Chunk:\n{chunk}"
    ))

    def as_messages(self, text: str) -> list:
        return [
            {"role": "system", "content": self.system_message},
            {"role": "user", "content": self.instruction_template.format(chunk=text.strip())},
        ]


@dataclass
class LowInfoPrompt(BasePrompt):
    """Prompt to decide whether a chunk contains useful information for RAG."""

    system_message: str = field(default=(
        "You are a content quality filter for a RAG system. "
        "Decide whether a text chunk contains useful, substantive information. "
        "Answer YES or NO only."
    ))

    instruction_template: str = field(default=(
        "Does this chunk contain useful information for answering questions?\n\n"
        "Answer NO if it is only:\n"
        "- A title/heading with no content\n"
        "- A table of contents or page number\n"
        "- Boilerplate (e.g. 'All rights reserved')\n\n"
        "Answer YES if it contains facts, arguments, explanations, or data.\n\n"
        "Chunk:\n{chunk}\n\n"
        "Answer:"
    ))

    def as_messages(self, text: str) -> list:
        return [
            {"role": "system", "content": self.system_message},
            {"role": "user", "content": self.instruction_template.format(chunk=text.strip())},
        ]
