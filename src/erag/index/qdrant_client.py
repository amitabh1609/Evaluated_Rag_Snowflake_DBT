"""Qdrant collection management — create, upsert, and query dense vectors.

Collection schema:
  - Single dense vector field (BGE-large, 1024-dim, cosine distance)
  - Payload: doc_id, chunk_id, text, title, url, source, chunk_index
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
)

from erag.index.embed import EMBEDDING_DIM

if TYPE_CHECKING:
    import numpy as np
    from erag.chunk import Chunk

logger = logging.getLogger(__name__)

VECTOR_FIELD = "dense"


def get_client(url: str = "http://localhost:6333", api_key: str = "") -> QdrantClient:
    return QdrantClient(url=url, api_key=api_key or None)


def collection_exists(client: QdrantClient, name: str) -> bool:
    return any(c.name == name for c in client.get_collections().collections)


def create_collection(client: QdrantClient, name: str, vector_size: int = EMBEDDING_DIM) -> None:
    if collection_exists(client, name):
        logger.info("collection '%s' already exists", name)
        return
    client.create_collection(
        collection_name=name,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )
    logger.info("created collection '%s' (dim=%d)", name, vector_size)


def drop_collection(client: QdrantClient, name: str) -> None:
    if collection_exists(client, name):
        client.delete_collection(name)
        logger.info("dropped collection '%s'", name)


def upsert_batch(
    client: QdrantClient,
    collection: str,
    chunks: list["Chunk"],
    vectors: "np.ndarray",
) -> None:
    """Upsert a batch of chunks with their dense vectors."""
    points = [
        PointStruct(
            id=abs(hash(c.chunk_id)) % (2**63),  # stable int id from chunk_id
            vector=vectors[i].tolist(),
            payload={
                "doc_id": c.doc_id,
                "chunk_id": c.chunk_id,
                "text": c.text,
                "chunk_index": c.chunk_index,
                "total_chunks": c.total_chunks,
                **c.metadata,
            },
        )
        for i, c in enumerate(chunks)
    ]
    client.upsert(collection_name=collection, points=points, wait=True)


def count_points(client: QdrantClient, collection: str) -> int:
    return client.count(collection_name=collection).count
