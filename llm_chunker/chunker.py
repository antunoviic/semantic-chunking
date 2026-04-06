from __future__ import annotations

import re
from typing import Optional

from .llm_client import QwenClient
from .prompts import ChunkingPrompt


class LLMChunker:
    """
    Splits text into semantically coherent chunks using an LLM.

    The LLM decides where meaningful boundaries lie — no fixed character
    or token count is imposed upfront.
    """

    def __init__(
        self,
        client: Optional[QwenClient] = None,
        prompt: Optional[ChunkingPrompt] = None,
    ) -> None:
        self.client = client or QwenClient()
        self.prompt = prompt or ChunkingPrompt()

    def chunk(self, text: str) -> list:
        """Return a list of semantic chunks for the given text."""
        messages = self.prompt.as_messages(text)
        raw = self.client.chat(messages)
        return self._parse_response(raw)

    def _parse_response(self, raw: str) -> list:
        """Split the model response on the delimiter defined in the prompt."""
        delimiter = self.prompt.delimiter
        chunks = raw.split(delimiter)
        return [c.strip() for c in chunks if c.strip()]
