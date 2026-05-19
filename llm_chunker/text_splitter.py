from __future__ import annotations

import re
from typing import Optional

import nltk
from nltk.tokenize import sent_tokenize

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
    for resource, pkg in [("tokenizers/punkt", "punkt"), ("tokenizers/punkt_tab", "punkt_tab")]:
        try:
            nltk.data.find(resource)
        except (LookupError, OSError):
            nltk.download(pkg, quiet=True)


def _detect_language(text: str) -> str:
    try:
        from langdetect import detect
        code = detect(text[:2000])
        language = _LANG_MAP.get(code, "english")
        print(f"[text_splitter] Detected language: {code} -> using '{language}'")
        return language
    except Exception:
        return "english"


class TextSplitter:
    """
    Splits raw text into mini-chunks (sentence groups).
    Language is auto-detected unless explicitly provided.
    """

    def __init__(self, sentences_per_chunk: int = 3, language: Optional[str] = None) -> None:
        self._sentences_per_chunk = sentences_per_chunk
        self._language = language
        _ensure_nltk_data()

    def make_mini_chunks(self, text: str) -> list[str]:
        language = self._language or _detect_language(text)
        paragraphs = self._paragraph_split(text)
        mini_chunks = []
        for para in paragraphs:
            sentences = sent_tokenize(para, language=language)
            if len(sentences) <= self._sentences_per_chunk:
                mini_chunks.append(para)
            else:
                for i in range(0, len(sentences), self._sentences_per_chunk):
                    group = sentences[i:i + self._sentences_per_chunk]
                    mini_chunks.append(" ".join(group))
        return mini_chunks

    @staticmethod
    def _paragraph_split(text: str) -> list[str]:
        normalized = text.replace('\r\n', '\n').replace('\r', '\n')
        normalized = re.sub(r'\n{2,}', '\n\n', normalized)
        parts = re.split(r'\n\n', normalized)
        return [p.strip() for p in parts if p.strip()]
