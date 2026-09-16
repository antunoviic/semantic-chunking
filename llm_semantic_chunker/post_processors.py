from __future__ import annotations

import re

from .interfaces import BasePrompt, ChunkPostProcessor, LLMClient
from ._logging import get_logger

logger = get_logger(__name__)


class LowInfoFilter(ChunkPostProcessor):
    """Removes chunks the LLM judges as low-information."""

    def __init__(self, client: LLMClient, prompt: BasePrompt, verbose: bool = False) -> None:
        self.client = client
        self.prompt = prompt
        self.verbose = verbose

    # No/Nein as filter
    _EXPLICIT_NO = re.compile(r"^\W*(NO|NEIN)\b", re.IGNORECASE)

    def process(self, chunks: list[str]) -> list[str]:
        result = []
        for chunk in chunks:
            messages = self.prompt.as_messages(chunk)
            raw = self.client.chat(messages).strip()
            drop = bool(self._EXPLICIT_NO.match(raw))
            logger.debug(f"[filter] {'removed' if drop else 'kept':<7} raw={raw[:40]!r} "
                  f"-> {chunk[:60]}...")
            if not drop:
                result.append(chunk)
        logger.debug(f"[LowInfoFilter] {len(chunks)} -> {len(result)} chunks")
        return result


class ChunkEnricher(ChunkPostProcessor):

    def __init__(self, client: LLMClient, prompt: BasePrompt, verbose: bool = False) -> None:
        self.client = client
        self.prompt = prompt
        self.verbose = verbose

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
