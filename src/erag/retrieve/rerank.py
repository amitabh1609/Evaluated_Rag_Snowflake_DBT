"""Cross-encoder reranking using BAAI/bge-reranker-large.

A cross-encoder jointly encodes (query, passage) pairs and outputs a
relevance score, giving much higher precision than bi-encoder similarity
alone. Most RAG tutorials skip this step entirely.

The model is lazy-loaded and cached as a module-level singleton to avoid
reloading across calls in the same process.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sentence_transformers import CrossEncoder

from erag.retrieve.hybrid import RankedResult

logger = logging.getLogger(__name__)

_reranker: "CrossEncoder | None" = None


def get_reranker(model_name: str = "BAAI/bge-reranker-large") -> "CrossEncoder":
    global _reranker
    if _reranker is None:
        from sentence_transformers import CrossEncoder
        logger.info("loading reranker %s …", model_name)
        _reranker = CrossEncoder(model_name, max_length=512)
    return _reranker


def rerank(
    query: str,
    candidates: list[RankedResult],
    top_k: int,
    model_name: str = "BAAI/bge-reranker-large",
) -> list[RankedResult]:
    """Rerank `candidates` for `query` using a cross-encoder.

    Args:
        query: The (rewritten) search query.
        candidates: RRF-merged candidates from hybrid retrieval.
        top_k: Number of results to return after reranking.
        model_name: HuggingFace model identifier.

    Returns:
        Top `top_k` candidates sorted by cross-encoder score (descending).
        The original RRF score is preserved in metadata["rrf_score"].
        The cross-encoder score becomes the primary `score` field.
    """
    if not candidates:
        return []

    model = get_reranker(model_name)
    pairs = [(query, c.metadata.get("text", "")) for c in candidates]
    scores = model.predict(pairs, show_progress_bar=False)

    reranked = []
    for cand, ce_score in zip(candidates, scores):
        meta = {**cand.metadata, "rrf_score": cand.score, "rerank_score": float(ce_score)}
        reranked.append(
            RankedResult(
                doc_id=cand.doc_id,
                chunk_id=cand.chunk_id,
                score=float(ce_score),
                metadata=meta,
            )
        )

    reranked.sort(key=lambda r: r.score, reverse=True)
    return reranked[:top_k]
