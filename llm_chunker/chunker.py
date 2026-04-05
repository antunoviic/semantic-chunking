from __future__ import annotations

import json
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
        """Extract a JSON string array from the model response."""
        # Strip markdown code fences if present
        cleaned = re.sub(r"```(?:json)?\s*|\s*```", "", raw).strip()

        try:
            chunks = json.loads(cleaned)
        except json.JSONDecodeError:
            # Fallback: try to find the first JSON array in the response
            match = re.search(r"\[.*\]", cleaned, re.DOTALL)
            if match:
                chunks = json.loads(match.group())
            else:
                raise ValueError(
                    f"Could not parse LLM response as JSON array.\nRaw response:\n{raw}"
                )

        if not isinstance(chunks, list) or not all(isinstance(c, str) for c in chunks):
            raise TypeError("Expected a JSON array of strings from the LLM.")

        return [c.strip() for c in chunks if c.strip()]
