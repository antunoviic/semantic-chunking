from __future__ import annotations

from dataclasses import dataclass, field

from ..interfaces import BasePrompt


@dataclass
class BoundaryPrompt(BasePrompt):
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
        "Return ONLY a comma-separated list of chunk numbers where a new topic STARTS.\n"
        "Example: 3, 7\n"
        "If there is no topic change, respond with exactly: NONE\n"
        "Do NOT explain your answer. Do NOT use any other words.\n\n"
        "Segments:\n{tagged_text}"
    ))

    def as_messages(self, text: str) -> list:
        return [
            {"role": "system", "content": self.system_message},
            {"role": "user", "content": self.instruction_template.format(tagged_text=text)},
        ]
