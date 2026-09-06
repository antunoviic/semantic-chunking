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


@dataclass
class SplitPointPrompt:
    """
    instead of a hard cut at the end, the LLM picks the single most sensible boundary WITHIN the accumulated
    sentences.
    """

    system_message: str = field(default=(
        "You are a text segmentation tool. Given a list of consecutive sentences "
        "that has grown too long to keep as one chunk, you find the single best "
        "place to split it into two topically coherent parts."
    ))

    instruction_template: str = field(default=(
        "These {n} sentences must be split into two parts because the chunk is too "
        "long. Choose the ONE boundary where the topic shift is strongest, so the "
        "first part is as self-contained as possible.\n\n"
        "{numbered}\n\n"
        "Answer with ONLY the number of the sentence that STARTS the second part "
        "(a single integer from 2 to {n}). No other text."
    ))

    def as_messages(self, sentences: list) -> list:
        numbered = "\n".join(f"[{i}] {s}" for i, s in enumerate(sentences, 1))
        return [
            {"role": "system", "content": self.system_message},
            {"role": "user", "content": self.instruction_template.format(
                n=len(sentences), numbered=numbered
            )},
        ]
