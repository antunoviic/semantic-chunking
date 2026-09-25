from __future__ import annotations

import ast
import functools
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional


_CODE_ROOTS = (
    "llm_semantic_chunker/chunker.py",
    "llm_semantic_chunker/config.py",
    "llm_semantic_chunker/text_splitter.py",
    "llm_semantic_chunker/post_processors.py",
    "llm_semantic_chunker/prompts.py",
    "llm_semantic_chunker/llm_client.py",
    "llm_semantic_chunker/incremental",
    "llm_semantic_chunker/window",
    "app/document_reader.py",
)


#   The interpreter that produced a cache. `ast.dump()` is not stable across
#   Python minor versions — new node fields and changed defaults alter its text
#   for unchanged source — so a digest is only comparable within one version.
#   It is stamped separately rather than hashed in, so that a mismatch can be
#   reported as what it is instead of looking like a code change.
_PY_TAG = f"{sys.version_info.major}.{sys.version_info.minor}"


@functools.lru_cache(maxsize=None)
def _code_digest() -> str:
    """Fingerprint of the code that draws chunk boundaries (AST, no docstrings).

    Comparable only within one Python minor version; see _PY_TAG. Cached per
    process: the files do not change while a run is going, and every cache load
    compares against it.
    """
    h = hashlib.sha256()
    for root in _CODE_ROOTS:
        p = Path(root)
        files = sorted(p.rglob("*.py")) if p.is_dir() else ([p] if p.is_file() else [])
        for f in files:
            if "__pycache__" in f.parts:
                continue
            h.update(f.as_posix().encode())
            h.update(_semantic_source(f).encode())
    return h.hexdigest()[:12]


def _semantic_source(path: Path) -> str:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return ""
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return text
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                             ast.AsyncFunctionDef)) and ast.get_docstring(node):
            node.body = node.body[1:] or [ast.Pass()]
    return ast.dump(tree)


class ChunkCache:
    """SRP: loads and saves chunk JSON caches with automatic backup."""

    def __init__(self, cache_dir: Path = Path("./chunks_cache")) -> None:
        self._dir = cache_dir

    def _path(self, source_path: str, enriched: bool, variant: str = "") -> Path:
        suffix = (f"_{variant}" if variant else "") + ("_enriched" if enriched else "")
        return self._dir / (Path(source_path).stem + suffix + ".json")

    def load(self, source_path: str, enriched: bool = False, variant: str = "",
             allow_stale: bool = False) -> Optional[list[str]]:
        """Chunks of one arm, or None if there is no usable cache.

        A cache whose `code_digest` differs from the code running now is treated
        as absent: its boundaries were drawn by other code, and loading it next
        to fresh arms would put two code states into one report. Unstamped
        caches count as stale. `allow_stale=True` loads such a cache anyway,
        for inspection, never for a comparison.

        A different Python minor version produces a different digest for
        unchanged code, so the message names that case separately — otherwise a
        switched interpreter is indistinguishable from an edited chunker.
        """
        path = self._path(source_path, enriched, variant)
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        have, want = data.get("code_digest"), _code_digest()
        if have != want and not allow_stale:
            print(f"[cache] Skipped {path.name}: chunked with code {have or 'unstamped'} "
                  f"on {str(data.get('chunked_at', '?'))[:16]}, current code is {want}. "
                  f"Re-chunk it (--rechunk) or load with allow_stale=True.")
            was = data.get("python")
            if was and was != _PY_TAG:
                print(f"        Note: this cache was written under Python {was}, you are "
                      f"running {_PY_TAG}. The digest covers the AST, whose text form "
                      f"changes between minor versions, so the chunker itself may be "
                      f"unchanged. Run under Python {was} to reuse it.")
            return None
        print(f"[cache] Loaded {len(data['chunks'])} chunks from {path.name}")
        print(f"        Chunked on: {data['chunked_at']}   code: {have or 'unstamped'}"
              + ("   (STALE, loaded on request)" if have != want else ""))
        return data["chunks"]

    def provenance(self, source_path: str, enriched: bool = False,
                   variant: str = "") -> Optional[dict]:
        """Header data without the chunks — for reports that need to show that all
        compared arms came from the same code state."""
        path = self._path(source_path, enriched, variant)
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        return {k: v for k, v in data.items() if k != "chunks"}

    def save(self, source_path: str, chunks: list[str], enriched: bool = False,
             variant: str = "", params: Optional[dict] = None) -> None:
        self._dir.mkdir(exist_ok=True)
        path = self._path(source_path, enriched, variant)
        if path.exists():
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup = path.with_suffix(f".{ts}.json")
            path.rename(backup)
            print(f"[cache] Old chunks backed up to {backup.name}")
        #   chunked_at   — when
        #   code_digest  — which code (see _code_digest)
        #   python       — which interpreter; the digest is only comparable
        #                  within one minor version
        #   params       — ChunkerConfig plus, from the pipeline, the document
        #                  hash, the LLM settings and the library versions
        data = {
            "chunked_at":  datetime.now().isoformat(),
            "code_digest": _code_digest(),
            "python":      _PY_TAG,
            "params":      params or {},
            "chunks":      chunks,
        }
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[cache] Saved {len(chunks)} chunks to {path.name}")
