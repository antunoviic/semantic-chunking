from __future__ import annotations

import re
from typing import Optional

import nltk
from nltk.tokenize import sent_tokenize
from ._logging import get_logger

logger = get_logger(__name__)

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
        import langdetect
        # langdetect fixed seed to be deterministic
        langdetect.DetectorFactory.seed = 0
        code = langdetect.detect(text[:2000])
        language = _LANG_MAP.get(code, "english")
        logger.debug(f"[text_splitter] Detected language: {code} -> using '{language}'")
        return language
    except Exception:
        return "english"


class TextSplitter:

    #Language is auto-detected unless explicitly provided.


    def __init__(self, sentences_per_chunk: int = 3, language: Optional[str] = None) -> None:
        self._sentences_per_chunk = sentences_per_chunk
        self._language = _LANG_MAP.get(language, language) if language else None
        self._detected: Optional[str] = None
        _ensure_nltk_data()

    def _resolve_language(self, text: str) -> str:
        if self._language:
            return self._language
        if self._detected is None:
            self._detected = _detect_language(text)
        return self._detected

    def split_sentences(self, text: str) -> list[str]:
        #individual sentences for incremental
        language = self._resolve_language(text)
        sentences = []
        for para in self._paragraph_split(text):
            sentences.extend(self._tokenize(para, language))
        return sentences

    @staticmethod
    def _tokenize(paragraph: str, language: str) -> list[str]:
        try:
            return sent_tokenize(paragraph, language=language)
        except LookupError as exc:
            raise ValueError(
                f"NLTK has no sentence tokeniser for language={language!r}. "
                f"Supported here: {', '.join(sorted(set(_LANG_MAP.values())))}. "
                f"Pass one of those as `language`, or leave it None to auto-detect."
            ) from exc

    def make_mini_chunks(self, text: str) -> list[str]:
        language = self._resolve_language(text)
        paragraphs = self._paragraph_split(text)
        mini_chunks = []
        for para in paragraphs:
            sentences = self._tokenize(para, language)
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
