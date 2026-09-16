from __future__ import annotations

from .interfaces import BasePrompt, ChunkPostProcessor, LLMClient
from .prompts import is_explicit_no
from ._logging import get_logger

logger = get_logger(__name__)


class LowInfoFilter(ChunkPostProcessor):
    """Removes chunks the LLM judges as low-information."""

    def __init__(self, client: LLMClient, prompt: BasePrompt) -> None:
        self.client = client
        self.prompt = prompt

    def process(self, chunks: list[str]) -> list[str]:
        result = []
        for chunk in chunks:
            messages = self.prompt.as_messages(chunk)
            raw = self.client.chat(messages).strip()
            drop = is_explicit_no(raw)
            logger.debug(f"[filter] {'removed' if drop else 'kept':<7} raw={raw[:40]!r} "
                  f"-> {chunk[:60]}...")
            if not drop:
                result.append(chunk)
        logger.debug(f"[LowInfoFilter] {len(chunks)} -> {len(result)} chunks")
        return result


class ChunkEnricher(ChunkPostProcessor):
    """Prefixes every chunk with an LLM-generated `[Topic: ...]` line."""

    def __init__(self, client: LLMClient, prompt: BasePrompt) -> None:
        self.client = client
        self.prompt = prompt

    def process(self, chunks: list[str]) -> list[str]:
        enriched = []
        for i, chunk in enumerate(chunks):
            messages = self.prompt.as_messages(chunk)
            response = self.client.chat(messages).strip()
            topic = self._parse_response(response)

            enriched_chunk = f"[Topic: {topic}]\n\n{chunk}" if topic else chunk

            logger.debug(f"[ChunkEnricher] [{i+1}/{len(chunks)}] Topic: {topic}")
            enriched.append(enriched_chunk)
        return enriched

    @staticmethod
    def _parse_response(response: str) -> str:
        for line in response.splitlines():
            line = line.strip()
            if line.lower().startswith("topic:"):
                return line[len("topic:"):].strip()
        return ""
