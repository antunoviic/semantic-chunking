import logging as _logging

# null to bypass handler of last resort
_logging.getLogger(__name__).addHandler(_logging.NullHandler())

try:                                      # single source: the installed metadata
    from importlib.metadata import PackageNotFoundError as _NotFound
    from importlib.metadata import version as _get_version
    __version__ = _get_version("llm-semantic-chunker")
except (ImportError, _NotFound):          # running from a source tree, not installed
    __version__ = "0+unknown"

from .chunker import LLMChunker
from .config import ChunkerConfig
from .llm_client import OllamaClient
from .incremental import IncrementalBoundaryDetector, IncrementalBoundaryPrompt
from .window import BoundaryDetector, BoundaryPrompt
from .interfaces import ChunkPostProcessor, LLMClient
from .prompts import EnrichmentPrompt, LowInfoPrompt

__all__ = [
    "LLMChunker",
    "ChunkerConfig",
    "OllamaClient",
    "LLMClient",
    "ChunkPostProcessor",
    "IncrementalBoundaryDetector",
    "IncrementalBoundaryPrompt",
    "BoundaryDetector",
    "BoundaryPrompt",
    "EnrichmentPrompt",
    "LowInfoPrompt",
]
