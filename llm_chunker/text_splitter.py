from __future__ import annotations

import re
from typing import Optional

import nltk
from nltk.tokenize import sent_tokenize

# Maps langdetect language codes -> NLTK language names
_LANG_MAP = {
    "en": "english",
    "de": "german",
    "fr": "french",
    "es": "spanish",
    "it": "italian",
    "pt": "portuguese",
    "nl": "dutch",
}


def _ensure_nltk_data() -> None:
    """Download the Punkt tokenizer data if not already present."""
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        nltk.download("punkt_tab", quiet=True)


def _detect_language(text: str) -> str:
    """
    Auto-detect the language of the text and return the NLTK language name.
    Falls back to 'english' if detection fails or language is unsupported.
    """
    try:
        from langdetect import detect
        code = detect(text[:2000])  # first 2000 chars are enough
        language = _LANG_MAP.get(code, "english")
        print(f"[text_splitter] Detected language: {code} -> using '{language}'")
        return language
    except Exception:
        return "english"


class TextSplitter:
    """
    Splits raw text into mini-chunks (sentence groups) for the
    sliding-window boundary detection stage.

    Uses NLTK's Punkt tokenizer -- no manual abbreviation lists needed,
    supports multiple languages. Language is auto-detected unless provided.

    Parameters
    ----------
    sentences_per_chunk : int
        How many sentences to group into one mini-chunk.
    language : str or None
        NLTK language name (e.g. "english", "german").
        If None, the language is auto-detected from the input text.
    """

    def __init__(self, sentences_per_chunk: int = 3, language: Optional[str] = None) -> None:
        self.sentences_per_chunk = sentences_per_chunk
        self._language = language  # None = auto-detect on first call
        _ensure_nltk_data()

    def make_mini_chunks(self, text: str) -> list[str]:
        language = self._language or _detect_language(text)
        paragraphs = self._paragraph_split(text)
        mini_chunks = []
        for para in paragraphs:
            sentences = self._sentence_split(para, language)
            if len(sentences) <= self.sentences_per_chunk:
                mini_chunks.append(para)
            else:
                for i in range(0, len(sentences), self.sentences_per_chunk):
                    group = sentences[i:i + self.sentences_per_chunk]
                    mini_chunks.append(" ".join(group))
        return mini_chunks

    @staticmethod
    def _paragraph_split(text: str) -> list[str]:
        normalized = text.replace('\r\n', '\n').replace('\r', '\n')
        normalized = re.sub(r'\n{2,}', '\n\n', normalized)
        parts = re.split(r'\n\n', normalized)
        return [p.strip() for p in parts if p.strip()]

    @staticmethod
    def _sentence_split(text: str, language: str) -> list[str]:
        return sent_tokenize(text, language=language)
