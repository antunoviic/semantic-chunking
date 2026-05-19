from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Optional


class ChunkCache:
    """SRP: loads and saves chunk JSON caches with automatic backup."""

    def __init__(self, cache_dir: Path = Path("./chunks_cache")) -> None:
        self._dir = cache_dir

    def _path(self, pdf_path: str, enriched: bool) -> Path:
        suffix = "_enriched" if enriched else ""
        return self._dir / (Path(pdf_path).stem + suffix + ".json")

    def load(self, pdf_path: str, enriched: bool = False) -> Optional[list[str]]:
        path = self._path(pdf_path, enriched)
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        print(f"[cache] Loaded {len(data['chunks'])} chunks from {path.name}")
        print(f"        Chunked on: {data['chunked_at']}")
        return data["chunks"]

    def save(self, pdf_path: str, chunks: list[str], enriched: bool = False) -> None:
        self._dir.mkdir(exist_ok=True)
        path = self._path(pdf_path, enriched)
        if path.exists():
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup = path.with_suffix(f".{ts}.json")
            path.rename(backup)
            print(f"[cache] Old chunks backed up to {backup.name}")
        data = {
            "source":      Path(pdf_path).name,
            "chunked_at":  datetime.now().isoformat(),
            "chunk_count": len(chunks),
            "chunks":      chunks,
        }
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[cache] Saved {len(chunks)} chunks to {path.name}")
