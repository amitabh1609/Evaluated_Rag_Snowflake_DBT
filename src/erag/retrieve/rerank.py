"""Cross-encoder reranking using BAAI/bge-reranker-large.

Takes the top-N RRF-merged candidates and reranks them with a cross-encoder
that jointly encodes (query, passage) pairs. Returns top_k_rerank results.
"""

# TODO Phase 2: implement
#   from sentence_transformers import CrossEncoder
#
#   def rerank(query: str, candidates: list[RankedResult], top_k: int) -> list[RankedResult]:
#       model = CrossEncoder("BAAI/bge-reranker-large")
#       scores = model.predict([(query, c.metadata["text"]) for c in candidates])
#       ...
