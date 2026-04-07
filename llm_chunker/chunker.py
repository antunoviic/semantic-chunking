from __future__ import annotations

from typing import Optional

from .llm_client import QwenClient
from .prompts import ChunkingPrompt, LowInfoPrompt


class LLMChunker:
    """
    Splits text into semantically coherent chunks using an LLM.

    Pipeline:
      1. Chunking   — LLM splits text at semantic boundaries
      2. Filtering  — LLM removes chunks with low information content
    """

    def __init__(
        self,
        client: Optional[QwenClient] = None,
        prompt: Optional[ChunkingPrompt] = None,
        low_info_prompt: Optional[LowInfoPrompt] = None,
        filter_low_info: bool = True,
    ) -> None:
        self.client = client or QwenClient()
        self.prompt = prompt or ChunkingPrompt()
        self.low_info_prompt = low_info_prompt or LowInfoPrompt()
        self.filter_low_info = filter_low_info

    def chunk(self, text: str) -> list[str]:
        """Return a list of semantic chunks for the given text."""
        raw = self.client.chat(self.prompt.as_messages(text))
        chunks = self._parse_response(raw)

        if self.filter_low_info:
            chunks = self._remove_low_info(chunks)

        return chunks

    def _parse_response(self, raw: str) -> list[str]:
        """Split the model response on the delimiter defined in the prompt."""
        parts = raw.split(self.prompt.delimiter)
        return [c.strip() for c in parts if c.strip()]

    def _remove_low_info(self, chunks: list[str]) -> list[str]:
        """Ask the LLM for each chunk whether it contains useful information."""
        result = []
        for chunk in chunks:
            messages = self.low_info_prompt.as_messages(chunk)
            response = self.client.chat(messages).strip().upper()
            if response.startswith("YES"):
                result.append(chunk)
            else:
                print(f"[filter] removed low-info chunk: {chunk[:60]}...")
        return result
