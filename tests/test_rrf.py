"""Unit tests for reciprocal rank fusion (pure function, no I/O)."""

from src.utils.rrf import reciprocal_rank_fusion


def test_rrf_rewards_items_ranked_in_both_lists():
    """An item present in both lists should outrank one present in only one."""
    dense = ["a", "b", "c"]
    sparse = ["b", "d", "a"]

    fused = reciprocal_rank_fusion([dense, sparse])

    # 'b' (rank 2 + rank 1) and 'a' (rank 1 + rank 3) both hit in both lists;
    # everything hit in only one list scores strictly lower.
    assert set(fused[:2]) == {"a", "b"}
    assert fused[-1] in {"c", "d"}


def test_rrf_preserves_single_list_order_when_only_one_list_given():
    ranked = ["x", "y", "z"]
    assert reciprocal_rank_fusion([ranked]) == ranked


def test_rrf_handles_empty_lists():
    assert reciprocal_rank_fusion([[], []]) == []
    assert reciprocal_rank_fusion([["only"], []]) == ["only"]


def test_rrf_dedupes_by_identity_not_position():
    """The same item appearing at different ranks in two lists is one entry, not two."""
    fused = reciprocal_rank_fusion([["a", "b"], ["b", "a"]])
    assert sorted(fused) == ["a", "b"]
    assert len(fused) == 2


def test_rrf_works_on_tuples_like_kg_chunk_pairs():
    """Real usage fuses (doc_id, content) tuples from vector/BM25 chunk retrieval."""
    dense = [(1, "chunk one"), (2, "chunk two")]
    sparse = [(2, "chunk two"), (3, "chunk three")]
    fused = reciprocal_rank_fusion([dense, sparse])
    assert fused[0] == (2, "chunk two")
    assert len(fused) == 3
