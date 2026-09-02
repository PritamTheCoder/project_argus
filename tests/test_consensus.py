"""
Project Argus - Consensus agent tests (pure functions)

The LLM judgement path is not exercised here (it requires a model); these tests
pin the deterministic clustering and quality-score logic the node relies on.
"""

from src.agents.consensus import cluster_facts, distinct_sources
from src.utils.quality import compute_quality_score


def test_cluster_groups_similar_embeddings():
    emb = [1.0] + [0.0] * 383
    other = [0.0, 1.0] + [0.0] * 382
    facts = [
        {"claim": "A", "embedding": emb},
        {"claim": "A2", "embedding": emb},
        {"claim": "B", "embedding": other},
    ]
    clusters = cluster_facts(facts)
    sizes = sorted(len(c) for c in clusters)
    assert sizes == [1, 2]


def test_cluster_no_embedding_is_singleton():
    facts = [{"claim": "A"}, {"claim": "B"}]
    clusters = cluster_facts(facts)
    assert len(clusters) == 2
    assert all(len(c) == 1 for c in clusters)


def test_distinct_sources_dedup_and_order():
    cluster = [
        {"source_url": "http://a"},
        {"source_url": "http://b"},
        {"source_url": "http://a"},
    ]
    assert distinct_sources(cluster) == ["http://a", "http://b"]


def test_quality_score_counts_and_credibility():
    verified = [
        {"support_level": "SUPPORTED", "credibility_score": 1.0, "source_url": "http://a", "corroboration_count": 2},
        {"support_level": "SUPPORTED", "credibility_score": 0.6, "source_url": "http://b", "corroboration_count": 1},
        {"support_level": "NOT_SUPPORTED", "credibility_score": 0.4, "source_url": "http://c"},
    ]
    qs = compute_quality_score(verified, source_map={}, contradiction_count=1)
    assert qs["verified_fact_count"] == 2
    assert qs["distinct_source_count"] == 2
    assert qs["avg_source_credibility"] == 0.8
    assert qs["corroborated_fact_count"] == 1
    assert qs["single_source_fact_count"] == 1
    assert qs["contradiction_count"] == 1


def test_quality_score_empty():
    qs = compute_quality_score([], source_map={}, contradiction_count=0)
    assert qs["verified_fact_count"] == 0
    assert qs["avg_source_credibility"] == 0.0


def test_distinct_sources_counts_merged_sources():
    """A claim several sites restated is backed by all of them, so the cluster
    is eligible for consensus judgement rather than looking single-source."""
    cluster = [
        {"claim": "A", "source_url": "http://a", "merged_sources": ["http://a", "http://b"]},
    ]
    assert sorted(distinct_sources(cluster)) == ["http://a", "http://b"]


def test_distinct_sources_deduplicates_across_merged_facts():
    cluster = [
        {"claim": "A", "source_url": "http://a", "merged_sources": ["http://a", "http://b"]},
        {"claim": "A2", "source_url": "http://b", "merged_sources": ["http://b", "http://c"]},
    ]
    assert sorted(distinct_sources(cluster)) == ["http://a", "http://b", "http://c"]
