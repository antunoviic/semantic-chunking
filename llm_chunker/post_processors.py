from __future__ import annotations

from .interfaces import BasePrompt, ChunkPostProcessor, LLMClient


class LowInfoFilter(ChunkPostProcessor):
    """Removes chunks the LLM judges as low-information."""

    def __init__(self, client: LLMClient, prompt: BasePrompt, verbose: bool = False) -> None:
        self.client = client
        self.prompt = prompt
        self.verbose = verbose

    def process(self, chunks: list[str]) -> list[str]:
        result = []
        for chunk in chunks:
            messages = self.prompt.as_messages(chunk)
            response = self.client.chat(messages).strip().upper()
            if response.startswith("YES"):
                result.append(chunk)
            else:
                print(f"[filter] removed: {chunk[:60]}...")
        if self.verbose:
            print(f"[LowInfoFilter] {len(chunks)} -> {len(result)} chunks")
        return result


class ChunkEnricher(ChunkPostProcessor):
    """
    Prepends a Topic prefix to each chunk before embedding.
    The topic reflects the chunk's structural position in the document
    (chapter/section heading, hierarchical if identifiable).

    Format:
        [Topic: 3. Stoic Virtues > 3.2 Justice]

        <original chunk text>
    """

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

            if self.verbose:
                print(f"[ChunkEnricher] [{i+1}/{len(chunks)}] Topic: {topic}")
            enriched.append(enriched_chunk)
        return enriched

    @staticmethod
    def _parse_response(response: str) -> str:
        for line in response.splitlines():
            line = line.strip()
            if line.lower().startswith("topic:"):
                return line[len("topic:"):].strip()
        return ""
