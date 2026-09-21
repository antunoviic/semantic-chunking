from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from llm_semantic_chunker import ChunkerConfig, OllamaClient
from llm_semantic_chunker.post_processors import ChunkEnricher
from llm_semantic_chunker.prompts import EnrichmentPrompt

from app.chunk_cache import ChunkCache, _code_digest
from app.pipeline.variant import cache_key, label

# configuration of run v4's reference arm
REFERENCE = ChunkerConfig(
    mode="incremental",
    step_sentences=2,
    max_chunk_sentences=100,
    max_chunk_chars=1200,
    respect_headings=False,
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="python thesis/scripts/enrich_arm.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("documents", nargs="+", type=Path,
                   help="the .txt/.pdf files whose cached arm should be enriched")
    p.add_argument("--dry-run", action="store_true",
                   help="report what would happen, make no LLM call and write nothing")
    p.add_argument("--force", action="store_true",
                   help="rebuild even if an enriched arm with the current digest exists")
    p.add_argument("--cache-dir", type=Path, default=Path("./chunks_cache"))
    return p.parse_args(argv)


def enrich_document(doc: Path, cache: ChunkCache, client: OllamaClient | None,
                    dry_run: bool, force: bool) -> str:
    """Build the enriched arm for one document. Returns a one-line verdict."""
    parent_key = cache_key(REFERENCE)
    child_key = cache_key(REFERENCE, enrich=True)

    if not force:
        done = cache.load(str(doc), variant=child_key)
        if done:
            return f"{doc.name}: already done ({len(done)} chunks as {child_key})"

    parent = cache.load(str(doc), variant=parent_key)
    if parent is None:
        return (f"{doc.name}: no usable parent arm. Expected "
                f"{doc.stem}_{parent_key}.json with digest {_code_digest()}; "
                "chunk it first or re-chunk it.")

    if dry_run:
        return (f"{doc.name}: would enrich {len(parent)} chunks "
                f"({parent_key} -> {child_key}), about {len(parent) * 2.65 / 60:.0f} min")

    started = time.time()
    enriched = ChunkEnricher(client, EnrichmentPrompt()).process(parent)
    if len(enriched) != len(parent):
        raise RuntimeError(
            f"{doc.name}: enrichment changed the chunk count "
            f"({len(parent)} -> {len(enriched)}); it must only prefix text.")

    params = dataclasses.asdict(dataclasses.replace(REFERENCE, enrich=True))
    params["derived_from"] = {
        "arm": parent_key,
        "cache": f"{doc.stem}_{parent_key}.json",
        "note": "boundaries inherited unchanged; only [Topic: ...] prefixes added",
    }
    parent_head = cache.provenance(str(doc), variant=parent_key) or {}
    for key in ("document", "llm", "environment", "boundary_stats"):
        if key in (parent_head.get("params") or {}):
            params[key] = parent_head["params"][key]
    params["enriched_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")

    cache.save(str(doc), enriched, variant=child_key, params=params)
    mins = (time.time() - started) / 60
    return (f"{doc.name}: {len(enriched)} chunks written as {child_key} "
            f"in {mins:.0f} min — appears as '{label(REFERENCE, enrich=True)}' in the report")


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    cache = ChunkCache(cache_dir=args.cache_dir)
    client = None if args.dry_run else OllamaClient()

    print(f"[enrich] code digest: {_code_digest()}")
    print(f"[enrich] parent arm : {cache_key(REFERENCE)}")
    print(f"[enrich] target arm : {cache_key(REFERENCE, enrich=True)}\n")

    failures = 0
    for doc in args.documents:
        if not doc.exists():
            print(f"  !! {doc}: file not found")
            failures += 1
            continue
        try:
            print(f"  {enrich_document(doc, cache, client, args.dry_run, args.force)}")
        except Exception as exc:                                  # noqa: BLE001
            print(f"  !! {doc.name}: {type(exc).__name__}: {exc}")
            failures += 1

    if not args.dry_run:
        print("\n[enrich] Re-run the evaluation to pick the arm up:")
        print("           PHASE=2 bash thesis/scripts/run_v4.sh")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
