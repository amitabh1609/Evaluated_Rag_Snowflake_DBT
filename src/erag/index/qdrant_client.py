"""Thin wrapper around qdrant-client.

Creates/manages a collection with both dense (BGE-large) and sparse (BM25) vectors.
"""

# TODO Phase 2: implement
#   from qdrant_client import QdrantClient
#   from qdrant_client.models import (
#       Distance, VectorParams, SparseVectorParams, SparseIndexParams
#   )
#
#   DENSE_DIM = 1024  # BGE-large-en-v1.5 output dimension
#
#   def get_client(url: str, api_key: str = "") -> QdrantClient: ...
#   def create_collection(client: QdrantClient, name: str) -> None: ...
#   def upsert_batch(client: QdrantClient, collection: str, points: list) -> None: ...
