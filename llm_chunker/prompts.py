from dataclasses import dataclass, field


@dataclass
class BoundaryPrompt:
    """
    Prompt for the sliding-window boundary detection.
    The LLM sees tagged mini-chunks and returns indices where topics change.
    """

    system_message: str = field(default=(
        "You are a text segmentation tool for RAG pipelines. "
        "You identify topic boundaries in text. You never alter the text itself."
    ))

    instruction_template: str = field(default=(
        "Below are numbered text segments from a document. "
        "Your job: identify where the TOPIC changes.\n\n"
        "Rules:\n"
        "1. A topic boundary = the subject, entity, or theme clearly shifts.\n"
        "2. Minor transitions (e.g. an example within the same topic) are NOT boundaries.\n"
        "3. Headings belong to the paragraph they introduce — do NOT split them.\n"
        "4. If there is no clear topic change, respond with: NONE\n\n"
        "Output format:\n"
        "Return ONLY the chunk numbers where a new topic STARTS, "
        "separated by commas. Example: 3, 7\n"
        "This means: chunk_3 starts a new topic, chunk_7 starts another.\n\n"
        "Segments:\n{tagged_text}"
    ))

    def as_messages(self, tagged_text: str) -> list:
        return [
            {"role": "system", "content": self.system_message},
            {"role": "user", "content": self.instruction_template.format(tagged_text=tagged_text)},
        ]


@dataclass
class LowInfoPrompt:
    """Prompt to decide whether a chunk contains useful information for RAG."""

    system_message: str = field(default=(
        "You are a content quality filter for a RAG system. "
        "Decide whether a text chunk contains useful, substantive information. "
        "Answer YES or NO only."
    ))

    instruction_template: str = field(default=(
        "Does this chunk contain useful information for answering questions?\n\n"
        "Answer NO if it is only:\n"
        "- A title/heading with no content\n"
        "- A table of contents or page number\n"
        "- Boilerplate (e.g. 'All rights reserved')\n\n"
        "Answer YES if it contains facts, arguments, explanations, or data.\n\n"
        "Chunk:\n{chunk}\n\n"
        "Answer:"
    ))

    def as_messages(self, chunk: str) -> list:
        return [
            {"role": "system", "content": self.system_message},
            {"role": "user", "content": self.instruction_template.format(chunk=chunk.strip())},
        ]