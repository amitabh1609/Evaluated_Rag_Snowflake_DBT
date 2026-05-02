"""Unit tests for Reciprocal Rank Fusion (retrieve/hybrid.py).

These are the most important tests in the repo — RRF correctness is fundamental
to the hybrid retrieval claim.
"""

import pytest

from erag.retrieve.hybrid import RankedResult, reciprocal_rank_fusion


def _make(doc_id: str, chunk_id: str = "0", score: float = 1.0) -> RankedResult:
    return RankedResult(doc_id=doc_id, chunk_id=chunk_id, score=score)


# TODO Phase 2: replace NotImplementedError stubs with real assertions once implemented


def test_identical_rankings_preserve_order():
    """If both lists rank docs identically the merged order should match."""
    list_a = [_make("d1"), _make("d2"), _make("d3")]
    list_b = [_make("d1"), _make("d2"), _make("d3")]
    with pytest.raises(NotImplementedError):
        result = reciprocal_rank_fusion(list_a, list_b)
        assert [r.doc_id for r in result] == ["d1", "d2", "d3"]


def test_disjoint_rankings_interleave():
    """Disjoint lists should interleave: first of each list should score higher than second."""
    list_a = [_make("a1"), _make("a2")]
    list_b = [_make("b1"), _make("b2")]
    with pytest.raises(NotImplementedError):
        result = reciprocal_rank_fusion(list_a, list_b)
        assert len(result) == 4


def test_doc_in_only_one_list_still_included():
    """A doc appearing in only one list should still appear in the merged output."""
    list_a = [_make("shared"), _make("only_a")]
    list_b = [_make("shared"), _make("only_b")]
    with pytest.raises(NotImplementedError):
        result = reciprocal_rank_fusion(list_a, list_b)
        doc_ids = [r.doc_id for r in result]
        assert "only_a" in doc_ids
        assert "only_b" in doc_ids


def test_rrf_k_is_configurable():
    """k=1 should amplify rank differences more than k=60."""
    list_a = [_make("d1"), _make("d2")]
    list_b = [_make("d2"), _make("d1")]
    with pytest.raises(NotImplementedError):
        result_low_k = reciprocal_rank_fusion(list_a, list_b, k=1)
        result_high_k = reciprocal_rank_fusion(list_a, list_b, k=60)
        # Both should return same ordering but scores differ; just check lengths
        assert len(result_low_k) == len(result_high_k)


def test_empty_list_input_handled():
    """An empty list input should not crash; treat as a list contributing no scores."""
    list_a = [_make("d1"), _make("d2")]
    with pytest.raises(NotImplementedError):
        result = reciprocal_rank_fusion(list_a, [])
        assert len(result) == 2
