from __future__ import annotations

import re
from dataclasses import dataclass, field


_NUMBERED = re.compile(r"^(\d+(?:\.\d+)*\.?)\s+(\S.*)$")
_NAMED = re.compile(
    r"^(Abstract|Introduction|Background|Related Work|Methodology|Methods?|"
    r"Materials|Results|Discussion|Conclusions?|References|Bibliography|"
    r"Acknowledgements?|Appendix\s+[A-Z]?|Summary|Overview|"
    r"Normative References|Informative References)\b.*$"
)
_MAX_HEADING_LEN = 90
_MIN_HEADING_LEN = 3


def _is_heading_line(line: str) -> bool:
    if line[:1] in (" ", "\t"):          # list points/headings are indented
        return False
    s = line.strip()
    if not (_MIN_HEADING_LEN <= len(s) <= _MAX_HEADING_LEN):
        return False
    if "...." in s or "…" in s:          # table of contents
        return False
    if s.endswith((".", ":", ",", ";")) and not _NUMBERED.match(s):
        return False
    toks = s.split()
    numeric = sum(1 for t in toks if re.fullmatch(r"[\d.,()%–-]+", t))
    if toks and numeric > len(toks) * 0.4:   # tabelle row
        return False
    m = _NUMBERED.match(s)
    if m:
        return bool(re.match(r"^[A-Z(]", m.group(2)))
    return bool(_NAMED.match(s))


def extract_headings(raw_text: str) -> list[tuple[int, str]]:
    out: list[tuple[int, str]] = []
    pos = 0
    for line in raw_text.splitlines(keepends=True):
        if _is_heading_line(line):
            out.append((pos, " ".join(line.split())))
        pos += len(line)
    return out


def _skeleton(s: str) -> tuple[str, list[int]]:
    out, pos = [], []
    for i, ch in enumerate(s):
        if ch.isascii() and ch.isalnum():
            out.append(ch.lower())
            pos.append(i)
    return "".join(out), pos


def heading_sentence_indices(raw_text: str, sentences: list[str]) -> set[int]:
    headings = extract_headings(raw_text)
    if not headings:
        return set()
    text_sk, text_pos = _skeleton(raw_text)
    heading_positions = sorted(hpos for hpos, _ in headings)

    result: set[int] = set()
    cursor = 0
    h_ptr = 0
    for idx, sent in enumerate(sentences):
        probe, _ = _skeleton(sent[:60])
        if not probe:
            continue
        found = text_sk.find(probe, cursor)
        if found == -1:
            continue
        cursor = found
        char_pos = text_pos[found]
        if h_ptr < len(heading_positions) and char_pos >= heading_positions[h_ptr] - 2:
            result.add(idx)
            h_ptr += 1
    return result


# sorts out normal text
def looks_like_heading_candidate(sentence: str) -> bool:
    s = sentence.strip()
    if not (_MIN_HEADING_LEN <= len(s) <= _MAX_HEADING_LEN):
        return False
    if len(s.split()) > 12:
        return False
    if "...." in s or "…" in s:                      
        return False
    if not re.match(r"^[A-Z0-9(\[]", s):           
        return False
    if re.fullmatch(r"\d+(?:\.\d+)*\.?", s):
        return True
    if s.endswith((".", "!", "?")) and not _NUMBERED.match(s):
        return False
    return True


@dataclass
class HeadingOnlyPrompt:

    system_message: str = field(default=(
        "You classify a single line of text from a document. You decide whether "
        "it is a section heading or ordinary running prose."
    ))

    instruction_template: str = field(default=(
        "Here is a line from a document:\n\n{candidate}\n\n"
        "Is this line a SECTION HEADING (a short title that introduces a new "
        "section, chapter, or subsection), or is it ordinary running prose?\n\n"
        "Headings are short titles like 'Introduction', '3.2. Error Handling', "
        "'Related Work', 'Appendix A'. Ordinary prose is a normal sentence that "
        "makes a statement, even if it is short.\n\n"
        "Answer with exactly one word: HEADING or PROSE."
    ))

    def as_messages(self, candidate: str) -> list:
        return [
            {"role": "system", "content": self.system_message},
            {"role": "user", "content": self.instruction_template.format(
                candidate=candidate.strip()
            )},
        ]

    @staticmethod
    def parse(raw: str) -> bool:
        """True = Ueberschrift. Unklare Antworten gelten als PROSE (konservativ:
        im Zweifel keine zusaetzliche Grenze setzen)."""
        return bool(re.search(r"\bHEADING\b", raw, re.IGNORECASE))
