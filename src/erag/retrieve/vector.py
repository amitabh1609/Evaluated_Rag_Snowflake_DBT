"""Dense vector retrieval via Qdrant ANN search (HNSW, cosine distance)."""

from __future__ import annotations

import logging

from qdrant_client import QdrantClient

from erag.retrieve.hybrid import RankedResult

logger = logging.getLogger(__name__)


def vector_search(
    query_vector: list[float],
    top_k: int,
    client: QdrantClient,
    collection: str,
) -> list[RankedResult]:
    """Return top_k results ordered by cosine similarity to query_vector."""
    hits = client.query_points(
        collection_name=collection,
        query=query_vector,
        limit=top_k,
        with_payload=True,
    ).points

    results = []
    for hit in hits:
        payload = hit.payload or {}
        results.append(
            RankedResult(
                doc_id=payload.get("doc_id", ""),
                chunk_id=payload.get("chunk_id", str(hit.id)),
                score=float(hit.score),
                metadata=payload,
            )
        )
    return results
