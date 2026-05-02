"""Reciprocal Rank Fusion (RRF) merging of vector and BM25 result lists.

Formula: score(d) = sum(1 / (k + rank_i(d))) for each ranked list i
Default k = 60 (Cormack et al. 2009; also the value used in many production systems).
k is configurable via settings.rrf_k.

This module is the most important in the retrieval stack — unit-tested in
tests/test_hybrid_rrf.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RankedResult:
    doc_id: str
    chunk_id: str
    score: float = 0.0
    metadata: dict = field(default_factory=dict)


def reciprocal_rank_fusion(
    *ranked_lists: list[RankedResult],
    k: int = 60,
) -> list[RankedResult]:
    """Merge N ranked lists using RRF.

    Args:
        *ranked_lists: Each list is already sorted by descending relevance score.
        k: RRF constant (default 60). Higher k reduces the impact of top ranks.

    Returns:
        Merged list sorted by descending RRF score.
    """
    # TODO Phase 2: implement
    #   scores: dict[str, float] = {}
    #   for ranked in ranked_lists:
    #       for rank, result in enumerate(ranked, start=1):
    #           key = f"{result.doc_id}:{result.chunk_id}"
    #           scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
    #   ...
    raise NotImplementedError("Implement in Phase 2")
