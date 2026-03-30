from dataclasses import dataclass, field


@dataclass
class ChunkingPrompt:
    """Builds the prompt sent to the LLM for semantic chunking decisions."""

    system_message: str = field(default=(
        "You are a text segmentation expert. "
        "Your task is to identify semantically coherent chunk boundaries in a given text. "
        "A chunk should represent a self-contained unit of meaning — a topic, argument, or concept. "
        "Do not summarize or alter the text."
    ))

    instruction_template: str = field(default=(
        "Split the following text into semantically coherent chunks.\n"
        "Return ONLY a JSON array of strings, where each string is one chunk.\n"
        "Do not include any explanation, just the JSON array.\n\n"
        "Text:\n{text}"
    ))

    def build_user_message(self, text: str) -> str:
        return self.instruction_template.format(text=text)

    def as_messages(self, text: str) -> list[dict]:
        return [
            {"role": "system", "content": self.system_message},
            {"role": "user", "content": self.build_user_message(text)},
        ]
