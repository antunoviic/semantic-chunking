"""Evaluation pipeline.

`Pipeline` is imported lazily (PEP 562): it pulls in the vector store, the
embedding client and the baseline splitters, i.e. the whole `eval` extra.
Importing a name that needs none of that — `EvalConfig`, or the sibling modules
`variant` and `config` — must not require those packages to be installed.
"""
from .config import EvalConfig

__all__ = ["Pipeline", "EvalConfig"]


def __getattr__(name: str):
    if name == "Pipeline":
        from .runner import Pipeline
        return Pipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
