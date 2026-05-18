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
    Prepends a Topic + Summary prefix to each chunk before embedding.

    Format:
        [Topic: Naval on Happiness]
        [Summary: Happiness is achieved by removing desire, not fulfilling it.]

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
            topic, summary = self._parse_response(response)

            if topic or summary:
                prefix_parts = []
                if topic:
                    prefix_parts.append(f"[Topic: {topic}]")
                if summary:
                    prefix_parts.append(f"[Summary: {summary}]")
                enriched_chunk = " ".join(prefix_parts) + "\n\n" + chunk
            else:
                enriched_chunk = chunk

            if self.verbose:
                print(f"[ChunkEnricher] [{i+1}/{len(chunks)}] Topic: {topic}")
            enriched.append(enriched_chunk)
        return enriched

    @staticmethod
    def _parse_response(response: str) -> tuple[str, str]:
        topic = ""
        summary = ""
        for line in response.splitlines():
            line = line.strip()
            if line.lower().startswith("topic:"):
                topic = line[len("topic:"):].strip()
            elif line.lower().startswith("summary:"):
                summary = line[len("summary:"):].strip()
        return topic, summary
