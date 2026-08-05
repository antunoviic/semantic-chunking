from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class IncrementalBoundaryPrompt:
    """
    Prompt for incremental chunking: the LLM decides whether candidate
    sentences continue the topic of the current chunk or start a new one.
    """

    system_message: str = field(default=(
        "You are a text segmentation tool. You decide whether new sentences "
        "continue the same topic as the current text or start a new topic."
    ))

    instruction_template: str = field(default=(
        "Current chunk:\n{current_chunk}\n\n"
        "Candidate sentences (next part):\n{candidate}\n\n"
        "Question: Do the candidate sentences continue the SAME topic as the current chunk?\n"
        "Answer with YES or NO only. No explanation."
    ))

    def as_messages(self, current_chunk: str, candidate: str) -> list:
        return [
            {"role": "system", "content": self.system_message},
            {"role": "user", "content": self.instruction_template.format(
                current_chunk=current_chunk.strip(), candidate=candidate.strip()
            )},
        ]
