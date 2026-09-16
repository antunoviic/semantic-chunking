from __future__ import annotations

import ast
import hashlib
import json
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


def _code_digest() -> str:
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

    def load(self, source_path: str, enriched: bool = False, variant: str = "") -> Optional[list[str]]:
        path = self._path(source_path, enriched, variant)
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        print(f"[cache] Loaded {len(data['chunks'])} chunks from {path.name}")
        print(f"        Chunked on: {data['chunked_at']}"
              f"   code: {data.get('code_digest', 'ungestempelt')}")
        return data["chunks"]

    def provenance(self, source_path: str, enriched: bool = False,
                   variant: str = "") -> Optional[dict]:
        """Kopfdaten ohne die Chunks — fuer Reports, die belegen sollen, dass alle
        Vergleichsarme aus demselben Code-Stand stammen."""
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
        #   code_digest  — which code
        #   params       — settings
        data = {
            "chunked_at":  datetime.now().isoformat(),
            "code_digest": _code_digest(),
            "params":      params or {},
            "chunks":      chunks,
        }
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[cache] Saved {len(chunks)} chunks to {path.name}")
