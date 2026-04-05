from dataclasses import dataclass, field


@dataclass
class ChunkingPrompt:
    """Builds the prompt sent to the LLM for semantic chunking decisions."""

    system_message: str = field(default=(
        "You are a text segmentation expert for RAG pipelines. "
        "Your task is to group text into semantically coherent chunks. "
        "Each chunk must cover ONE topic or concept and contain MULTIPLE related sentences — never split a single sentence into its own chunk. "
        "Merge sentences that belong to the same topic into one chunk. "
        "Do not summarize, rephrase, or alter the text in any way."
    ))

    instruction_template: str = field(default=(
        "Split the following text into semantically coherent chunks for a RAG system.\n"
        "Rules:\n"
        "- Each chunk covers exactly ONE topic and contains multiple related sentences.\n"
        "- NEVER make a single sentence its own chunk — always group related sentences together.\n"
        "- Do not split headings from the paragraph they introduce.\n"
        "- Return ONLY a valid JSON array of strings (no objects, no keys, no markdown).\n"
        "Example: [\"Heading and its full paragraph text...\", \"Next topic with all its sentences...\"]\n\n"
        "Text:\n{text}"
    ))

    def build_user_message(self, text: str) -> str:
        return self.instruction_template.format(text=text)

    def as_messages(self, text: str) -> list:
        return [
            {"role": "system", "content": self.system_message},
            {"role": "user", "content": self.build_user_message(text)},
        ]
