from dataclasses import dataclass, field


@dataclass
class ChunkingPrompt:
    """Builds the prompt sent to the LLM for semantic chunking decisions."""

    system_message: str = field(default=(
        "You are a text segmentation tool for RAG pipelines. "
        "You split text at topic boundaries. You never alter, summarize, or rephrase the text."
    ))

    instruction_template: str = field(default=(
        "Split this text into semantically coherent chunks.\n\n"
        "Rules:\n"
        "1. A chunk = all consecutive sentences about the SAME topic, entity, or concept.\n"
        "2. Start a new chunk ONLY when the subject clearly changes.\n"
        "3. Never put a single sentence alone — merge it with its neighbours.\n"
        "4. Keep headings attached to the paragraph they introduce.\n"
        "5. Preserve the original text exactly — no rewriting.\n\n"
        "Output format:\n"
        "Return chunks separated by the delimiter |||. No JSON, no markdown, no numbering.\n\n"
        "GOOD example:\n"
        "Machine learning is a subset of AI. It allows systems to learn from data.|||"
        "The Eiffel Tower was built in 1889. It stands 330 meters tall.\n\n"
        "BAD example (single-sentence chunks — NEVER do this):\n"
        "Machine learning is a subset of AI.|||It allows systems to learn from data.\n\n"
        "Text:\n{text}"
    ))

    delimiter: str = "|||"

    def build_user_message(self, text: str) -> str:
        return self.instruction_template.format(text=text)

    def as_messages(self, text: str) -> list:
        return [
            {"role": "system", "content": self.system_message},
            {"role": "user", "content": self.build_user_message(text)},
        ]