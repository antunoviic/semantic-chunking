from __future__ import annotations

import logging

_ROOT = "llm_semantic_chunker"


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def enable_console_logging(level: int = logging.INFO) -> None:
    logger = logging.getLogger(_ROOT)
    if not any(getattr(h, "_llm_chunker_console", False) for h in logger.handlers):
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(message)s"))
        handler._llm_chunker_console = True          # type: ignore[attr-defined]
        logger.addHandler(handler)
    logger.setLevel(level)
