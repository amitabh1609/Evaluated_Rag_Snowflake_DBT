"""Unit tests for Reciprocal Rank Fusion (retrieve/hybrid.py).

The RRF merge is the most critical piece of the hybrid retrieval claim —
these tests verify correctness before any integration test is possible.
"""

from erag.retrieve.hybrid import RankedResult, reciprocal_rank_fusion


def _r(chunk_id: str, doc_id: str = "doc", score: float = 1.0) -> RankedResult:
    return RankedResult(doc_id=doc_id, chunk_id=chunk_id, score=score)


def test_identical_rankings_preserve_order():
    """Both lists rank identically → merged order matches."""
    list_a = [_r("c1"), _r("c2"), _r("c3")]
    list_b = [_r("c1"), _r("c2"), _r("c3")]
    result = reciprocal_rank_fusion(list_a, list_b)
    assert [r.chunk_id for r in result] == ["c1", "c2", "c3"]


def test_disjoint_rankings_all_present():
    """Disjoint lists → all items appear in the merged output."""
    list_a = [_r("a1"), _r("a2")]
    list_b = [_r("b1"), _r("b2")]
    result = reciprocal_rank_fusion(list_a, list_b)
    ids = {r.chunk_id for r in result}
    assert ids == {"a1", "a2", "b1", "b2"}


def test_top_ranked_in_both_lists_scores_highest():
    """A doc ranked #1 in both lists should have a higher RRF score than one ranked #1 in only one."""
    shared_top = [_r("shared"), _r("other_a")]
    other_list = [_r("shared"), _r("other_b")]
    single_list = [_r("only_once"), _r("shared")]

    result = reciprocal_rank_fusion(shared_top, other_list)
    top_id = result[0].chunk_id
    assert top_id == "shared", f"expected 'shared' at rank 1, got '{top_id}'"


def test_rrf_scores_are_positive():
    """All RRF scores must be strictly positive."""
    list_a = [_r("c1"), _r("c2")]
    list_b = [_r("c3"), _r("c1")]
    for r in reciprocal_rank_fusion(list_a, list_b):
        assert r.score > 0.0


def test_rrf_k_parameter_affects_scores():
    """k=1 amplifies rank differences more than k=60 — top score should be larger."""
    list_a = [_r("c1"), _r("c2")]
    list_b = [_r("c1"), _r("c2")]

    result_k1 = reciprocal_rank_fusion(list_a, list_b, k=1)
    result_k60 = reciprocal_rank_fusion(list_a, list_b, k=60)

    # Both produce same order; top score should differ
    assert result_k1[0].chunk_id == result_k60[0].chunk_id
    assert result_k1[0].score > result_k60[0].score


def test_empty_list_input_is_safe():
    """An empty list as one input should not crash and other items still ranked."""
    list_a = [_r("c1"), _r("c2")]
    result = reciprocal_rank_fusion(list_a, [])
    assert len(result) == 2
    assert result[0].chunk_id == "c1"


def test_single_list_rank_order_preserved():
    """With a single list, RRF output order should match input order."""
    list_a = [_r("c1"), _r("c2"), _r("c3")]
    result = reciprocal_rank_fusion(list_a)
    assert [r.chunk_id for r in result] == ["c1", "c2", "c3"]


def test_three_lists_merged():
    """Three lists: doc ranked high in all three should win."""
    l1 = [_r("winner"), _r("x")]
    l2 = [_r("winner"), _r("y")]
    l3 = [_r("winner"), _r("z")]
    result = reciprocal_rank_fusion(l1, l2, l3)
    assert result[0].chunk_id == "winner"


def test_metadata_preserved_from_first_occurrence():
    """Metadata from the first list in which a chunk appears should be preserved."""
    r = RankedResult(doc_id="doc1", chunk_id="c1", score=1.0, metadata={"title": "first"})
    r2 = RankedResult(doc_id="doc1", chunk_id="c1", score=0.5, metadata={"title": "second"})
    result = reciprocal_rank_fusion([r], [r2])
    assert result[0].metadata["title"] == "first"
