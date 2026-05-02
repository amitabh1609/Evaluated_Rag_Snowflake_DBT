"""BM25 retrieval via Qdrant sparse vectors (or rank_bm25 fallback).

Sparse vectors are computed at index time using a simple TFIDF-like tokenizer
and stored in Qdrant's sparse vector field.
"""

# TODO Phase 2: implement
#   Primary path: Qdrant sparse vector search (SparseVector query)
#   Fallback: rank_bm25.BM25Okapi over an in-memory corpus dump
