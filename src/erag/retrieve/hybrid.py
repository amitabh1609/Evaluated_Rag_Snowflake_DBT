"""Reciprocal Rank Fusion (RRF) merging of vector and BM25 result lists.

Formula: score(d) = Σ  1 / (k + rank_i(d))  for each ranked list i
Default k = 60 (Cormack, Clarke & Buettcher 2009; standard in production hybrid search).
k is configurable — higher k reduces the impact of top-rank differences.

Unit-tested in tests/test_hybrid_rrf.py.
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
        *ranked_lists: Each list is sorted by descending relevance (rank 1 = best).
                       Lists may be disjoint or overlapping.
        k: RRF constant. k=60 is the empirically validated default.

    Returns:
        Merged list sorted by descending RRF score.  Preserves metadata from
        the first list in which each doc appears.
    """
    rrf_scores: dict[str, float] = {}
    first_seen: dict[str, RankedResult] = {}

    for ranked in ranked_lists:
        for rank, result in enumerate(ranked, start=1):
            key = result.chunk_id  # chunk_id is the primary unique key
            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (k + rank)
            if key not in first_seen:
                first_seen[key] = result

    return [
        RankedResult(
            doc_id=first_seen[key].doc_id,
            chunk_id=key,
            score=rrf_scores[key],
            metadata=first_seen[key].metadata,
        )
        for key in sorted(rrf_scores, key=rrf_scores.__getitem__, reverse=True)
    ]
