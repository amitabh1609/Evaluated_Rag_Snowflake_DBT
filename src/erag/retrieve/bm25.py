"""BM25 retrieval using an in-memory rank_bm25 index loaded from disk.

The BM25 index is built by build_index.py and serialised to
data/processed/bm25_<collection>.pkl alongside a .payload.json file
mapping chunk_id → {text, title, url, source, doc_id, …}.

BM25 shines for exact-match terms (Snowflake feature names, dbt command
flags, version numbers) where semantic similarity is noisy.
"""

from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path

from erag.index.embed import load_bm25, tokenize_bm25
from erag.retrieve.hybrid import RankedResult

logger = logging.getLogger(__name__)


@lru_cache(maxsize=4)
def _load(bm25_path: str):
    """Cache loaded BM25 index so repeat queries don't re-deserialise."""
    bm25, chunk_ids = load_bm25(Path(bm25_path))
    payload_path = Path(bm25_path).with_suffix(".payload.json")
    payload: dict[str, dict] = {}
    if payload_path.exists():
        with open(payload_path) as f:
            payload = json.load(f)
    return bm25, chunk_ids, payload


def bm25_search(
    query: str,
    top_k: int,
    bm25_path: Path,
) -> list[RankedResult]:
    """Return top_k results ranked by BM25 score for the given query string."""
    bm25, chunk_ids, payload = _load(str(bm25_path))
    tokens = tokenize_bm25(query)
    if not tokens:
        return []

    scores = bm25.get_scores(tokens)
    # argsort descending — take top_k non-zero results
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

    results = []
    for idx in top_indices:
        if scores[idx] <= 0:
            continue
        chunk_id = chunk_ids[idx]
        meta = payload.get(chunk_id, {})
        doc_id = meta.get("doc_id", chunk_id.split(":")[0] if ":" in chunk_id else chunk_id)
        results.append(
            RankedResult(
                doc_id=doc_id,
                chunk_id=chunk_id,
                score=float(scores[idx]),
                metadata=meta,
            )
        )
    return results
