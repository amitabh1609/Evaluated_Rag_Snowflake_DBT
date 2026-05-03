"""Build the Qdrant index + BM25 index from crawled documents.

Pipeline: manifest.jsonl → read docs → chunk → embed → upsert Qdrant + save BM25

Usage:
    uv run python -m erag.index.build_index                      # default (semantic chunker, Config C)
    uv run python -m erag.index.build_index --chunker fixed      # fixed 512-token (Config A / B)
    uv run python -m erag.index.build_index --smoke              # 10-doc CI subset
    uv run python -m erag.index.build_index --reset              # drop collection and rebuild
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from tqdm import tqdm

from erag.chunk import Chunk
from erag.config import PROCESSED_DIR, RAW_DIR, settings
from erag.index import embed as emb
from erag.index import qdrant_client as qc

logger = logging.getLogger(__name__)

EMBED_BATCH = 32   # chunks per embedding batch (memory vs speed)
UPSERT_BATCH = 64  # chunks per Qdrant upsert call

BM25_PATH = PROCESSED_DIR / "bm25_{collection}.pkl"


def _load_manifest(manifest_path: Path, limit: int | None = None) -> list[dict]:
    entries: list[dict] = []
    if not manifest_path.exists():
        logger.warning("manifest not found: %s", manifest_path)
        return entries
    with open(manifest_path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    if limit:
        entries = entries[:limit]
    return entries


def _read_doc_text(doc_entry: dict) -> str | None:
    source = doc_entry.get("source", "")
    doc_id = doc_entry["doc_id"]
    sub = {
        "snowflake": "snowflake",
        "dbt_docs": "dbt_docs",
        "dbt_discourse": "dbt_discourse",
    }.get(source, source)
    path = RAW_DIR / sub / f"{doc_id}.md"
    if not path.exists():
        logger.debug("missing file: %s", path)
        return None
    return path.read_text(encoding="utf-8")


def _chunk_doc(entry: dict, text: str, chunker: str) -> list[Chunk]:
    meta = {
        "title": entry.get("title", ""),
        "url": entry.get("url", ""),
        "source": entry.get("source", ""),
    }
    if chunker == "semantic":
        from erag.chunk.semantic import chunk_document
    else:
        from erag.chunk.fixed import chunk_document
    return chunk_document(doc_id=entry["doc_id"], text=text, metadata=meta)


def build(
    chunker: str = "semantic",
    collection: str | None = None,
    smoke: bool = False,
    reset: bool = False,
) -> None:
    collection = collection or settings.qdrant_collection
    bm25_path = Path(str(BM25_PATH).format(collection=collection))

    client = qc.get_client(settings.qdrant_url, settings.qdrant_api_key)

    if reset:
        qc.drop_collection(client, collection)

    qc.create_collection(client, collection)

    manifest = _load_manifest(RAW_DIR / "manifest.jsonl", limit=10 if smoke else None)
    if not manifest:
        logger.error("no documents in manifest — run 'make crawl' first")
        sys.exit(1)

    logger.info("building index: chunker=%s collection=%s docs=%d", chunker, collection, len(manifest))

    all_chunks: list[Chunk] = []
    chunk_payload: dict[str, dict] = {}  # chunk_id → payload for BM25

    # ── Chunk all docs ────────────────────────────────────────────────────────
    for entry in tqdm(manifest, desc="chunking", unit="doc"):
        text = _read_doc_text(entry)
        if not text:
            continue
        chunks = _chunk_doc(entry, text, chunker)
        all_chunks.extend(chunks)
        for c in chunks:
            chunk_payload[c.chunk_id] = {**c.metadata, "text": c.text, "doc_id": c.doc_id}

    logger.info("total chunks: %d", len(all_chunks))

    # ── Embed + upsert in batches ─────────────────────────────────────────────
    for start in tqdm(range(0, len(all_chunks), UPSERT_BATCH), desc="embedding+upserting", unit="batch"):
        batch = all_chunks[start : start + UPSERT_BATCH]
        texts = [c.text for c in batch]
        vectors = emb.embed_documents(texts, settings.embedding_model, batch_size=EMBED_BATCH)
        qc.upsert_batch(client, collection, batch, vectors)

    logger.info("Qdrant upsert complete — %d points", qc.count_points(client, collection))

    # ── BM25 index ────────────────────────────────────────────────────────────
    bm25_index = emb.build_bm25_index(all_chunks)
    chunk_ids = [c.chunk_id for c in all_chunks]
    emb.save_bm25(bm25_index, chunk_ids, bm25_path)

    # Save chunk payload for BM25 result hydration
    payload_path = bm25_path.with_suffix(".payload.json")
    with open(payload_path, "w") as f:
        json.dump(chunk_payload, f)
    logger.info("chunk payload saved → %s", payload_path)

    logger.info("index build complete ✓")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Build Qdrant + BM25 index")
    parser.add_argument("--chunker", choices=["fixed", "semantic"], default="semantic")
    parser.add_argument("--collection", default=None, help="override collection name")
    parser.add_argument("--smoke", action="store_true", help="index first 10 docs only (CI)")
    parser.add_argument("--reset", action="store_true", help="drop collection before rebuilding")
    args = parser.parse_args()

    build(chunker=args.chunker, collection=args.collection, smoke=args.smoke, reset=args.reset)
