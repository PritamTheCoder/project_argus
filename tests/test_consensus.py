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


# ── consensus_node: evidence-graph persistence ───────────────────────────────

from unittest.mock import MagicMock, patch
from src.agents.consensus import consensus_node, ConsensusBatch, ClusterJudgment


def _supported_fact(claim, url, embedding):
    return {
        "claim": claim, "source_url": url, "support_level": "SUPPORTED",
        "confidence": 0.9, "embedding": embedding,
    }


@patch("src.agents.consensus.get_llm_with_fallbacks")
@patch("src.graph.kg.kg_store")
def test_consensus_stores_findings_and_contradictions_in_kg(mock_kg, mock_get_llm):
    mock_kg.store_consensus_findings = MagicMock()
    mock_kg.store_contradictions = MagicMock()
    mock_get_llm.return_value.invoke.return_value = ConsensusBatch(judgments=[
        ClusterJudgment(index=0, relationship="CONSENSUS", statement="Agreed: 400 Wh/kg"),
    ])

    emb = [1.0] + [0.0] * 383
    state = {
        "verified_facts": [
            _supported_fact("400 Wh/kg", "http://a.com", emb),
            _supported_fact("~400 Wh/kg", "http://b.com", emb),
        ],
        "source_map": {}, "session_id": "sess-1",
    }

    result = consensus_node(state)

    assert len(result["consensus_findings"]) == 1
    mock_kg.store_consensus_findings.assert_called_once()
    findings_arg = mock_kg.store_consensus_findings.call_args.args[0]
    assert findings_arg[0]["statement"] == "Agreed: 400 Wh/kg"
    mock_kg.store_contradictions.assert_called_once_with([], session_id="sess-1")


@patch("src.graph.kg.kg_store")
def test_consensus_handles_no_candidate_clusters_without_crashing(mock_kg):
    """Fewer than 2 supported facts means no LLM call and nothing to judge —
    storage must still be attempted (harmlessly, with empty lists)."""
    mock_kg.store_consensus_findings = MagicMock()
    mock_kg.store_contradictions = MagicMock()

    state = {"verified_facts": [], "source_map": {}, "session_id": "sess-1"}
    result = consensus_node(state)

    assert result["consensus_findings"] == []
    assert result["contradictions"] == []
    mock_kg.store_consensus_findings.assert_called_once_with([], session_id="sess-1")
    mock_kg.store_contradictions.assert_called_once_with([], session_id="sess-1")
