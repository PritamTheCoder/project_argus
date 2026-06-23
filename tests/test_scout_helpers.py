"""
Project Argus - Scout pure-helper tests

Covers the chunk->document alignment used during parent-document retrieval.
The previous implementation keyed a dict by chunk *text*, so identical chunk
text appearing in two documents silently collapsed to a single (arbitrary)
source. These tests pin the corrected, collision-free behaviour.
"""

from src.agents.scout import _align_reranked_to_docs


def test_align_basic_order_preserved():
    candidate = [(1, "A"), (2, "B"), (3, "C")]
    ranked = ["C", "A"]
    assert _align_reranked_to_docs(ranked, candidate) == [(3, "C"), (1, "A")]


def test_align_duplicate_chunk_text_not_collapsed():
    # "A" appears in both doc 1 and doc 2 — they must remain distinct sources.
    candidate = [(1, "A"), (2, "A"), (3, "B")]
    ranked = ["B", "A"]
    aligned = _align_reranked_to_docs(ranked, candidate)
    # "B" -> doc 3, first "A" occurrence -> doc 1 (consumed once, not collapsed).
    assert aligned == [(3, "B"), (1, "A")]


def test_align_consumes_each_pair_once():
    candidate = [(1, "A"), (2, "A")]
    ranked = ["A", "A"]
    aligned = _align_reranked_to_docs(ranked, candidate)
    # Both distinct sources recovered — not two references to the same doc.
    assert sorted(doc_id for doc_id, _ in aligned) == [1, 2]


def test_align_ignores_unknown_chunks():
    candidate = [(1, "A")]
    ranked = ["does-not-exist", "A"]
    assert _align_reranked_to_docs(ranked, candidate) == [(1, "A")]


def test_align_empty():
    assert _align_reranked_to_docs([], [(1, "A")]) == []
