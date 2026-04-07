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


@dataclass
class LowInfoPrompt:
    """Prompt to decide whether a chunk contains useful information for RAG."""

    system_message: str = field(default=(
        "You are a content quality filter for a RAG system. "
        "Your job is to decide whether a text chunk contains useful, substantive information "
        "that would help answer questions. "
        "Answer with YES if the chunk is informative, or NO if it should be removed."
    ))

    instruction_template: str = field(default=(
        "Does this text chunk contain useful, substantive information for a RAG system?\n\n"
        "Remove it (answer NO) if it is:\n"
        "- A title, heading, or document metadata with no content\n"
        "- A table of contents or index entry\n"
        "- A page number, header, or footer\n"
        "- Pure boilerplate (e.g. 'All rights reserved', 'Page 1 of 10')\n\n"
        "Keep it (answer YES) if it contains actual facts, arguments, explanations, or data.\n\n"
        "Chunk:\n{chunk}\n\n"
        "Answer YES or NO only."
    ))

    def as_messages(self, chunk: str) -> list:
        return [
            {"role": "system", "content": self.system_message},
            {"role": "user", "content": self.instruction_template.format(chunk=chunk.strip())},
        ]