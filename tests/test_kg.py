import pytest
import os
import tempfile
import gc
from src.graph.kg import KnowledgeGraph

def test_kg_store_and_retrieve():
    # Use a temp directory so Windows file-lock issues are avoided
    tmp_dir = tempfile.mkdtemp()
    db_path = os.path.join(tmp_dir, "test_kg.db")

    kg = KnowledgeGraph(db_path)

    # Store a dummy fact
    fact = {
        "claim": "Test claim",
        "source_url": "http://test.com",
        "source_excerpt": "Test excerpt",
        "support_level": "SUPPORTED",
        "confidence": 0.9,
        "embedding": [0.1] * 384
    }

    kg.store_facts([fact])

    # Retrieve
    results = kg.retrieve_relevant_facts([0.1] * 384, k=1)

    assert len(results) == 1
    assert results[0]["claim"] == "Test claim"
    assert results[0]["support_level"] == "SUPPORTED"

    # Properly close the connection (sqlite-vec holds handles on Windows)
    kg.db.close()
    del kg
    gc.collect()

    # Best-effort cleanup; ignore if Windows still holds it
    try:
        os.remove(db_path)
        os.rmdir(tmp_dir)
    except PermissionError:
        pass


def _make_fact(claim, support="SUPPORTED", confidence=0.9):
    return {
        "claim": claim,
        "source_url": "http://test.com",
        "source_excerpt": "excerpt",
        "support_level": support,
        "confidence": confidence,
        "embedding": [0.1] * 384,
    }


def test_kg_session_scoping():
    """Scoped retrieval returns only the requested session; global retrieval sees all."""
    tmp_dir = tempfile.mkdtemp()
    db_path = os.path.join(tmp_dir, "test_kg_session.db")
    kg = KnowledgeGraph(db_path)

    kg.store_facts([_make_fact("Fact A")], session_id="sessA")
    kg.store_facts([_make_fact("Fact B")], session_id="sessB")

    emb = [0.1] * 384
    scoped = kg.retrieve_relevant_facts(emb, k=10, session_id="sessA")
    assert {r["claim"] for r in scoped} == {"Fact A"}

    full = kg.retrieve_relevant_facts(emb, k=10)
    assert {r["claim"] for r in full} == {"Fact A", "Fact B"}

    kg.db.close()
    del kg
    gc.collect()
    try:
        os.remove(db_path)
        os.rmdir(tmp_dir)
    except PermissionError:
        pass


def test_kg_docs_session_scoping():
    """retrieve_top_docs is scoped like facts: concurrent runs can't see each other's docs."""
    tmp_dir = tempfile.mkdtemp()
    db_path = os.path.join(tmp_dir, "test_kg_docs_session.db")
    kg = KnowledgeGraph(db_path)

    kg.store_document_and_chunks(
        url="http://a.com", query="q", summary="s", summary_embedding=[0.1] * 384,
        chunks=["chunk a"], chunk_embeddings=[[0.1] * 384], session_id="sessA",
    )
    kg.store_document_and_chunks(
        url="http://b.com", query="q", summary="s", summary_embedding=[0.1] * 384,
        chunks=["chunk b"], chunk_embeddings=[[0.1] * 384], session_id="sessB",
    )

    emb = [0.1] * 384
    scoped_ids = kg.retrieve_top_docs(emb, k=10, session_id="sessA")
    assert len(scoped_ids) == 1
    assert kg.get_doc_metadata(scoped_ids[0])["url"] == "http://a.com"

    full_ids = kg.retrieve_top_docs(emb, k=10)
    assert len(full_ids) == 2

    kg.db.close()
    del kg
    gc.collect()
    try:
        os.remove(db_path)
        os.rmdir(tmp_dir)
    except PermissionError:
        pass


def test_kg_bm25_chunk_retrieval():
    """Chunks are indexed into FTS5 alongside the vector index and are keyword-searchable."""
    tmp_dir = tempfile.mkdtemp()
    db_path = os.path.join(tmp_dir, "test_kg_bm25.db")
    kg = KnowledgeGraph(db_path)
    assert kg.fts_available, "FTS5 should be available on a standard sqlite3 build"

    chunks = [
        "The QSE-5 solid-state battery reaches 380 Wh/kg energy density.",
        "Market analysts expect commercial vehicle sales to grow in 2026.",
    ]
    doc_id = kg.store_document_and_chunks(
        url="http://test.com/battery",
        query="battery energy density",
        summary="battery summary",
        summary_embedding=[0.1] * 384,
        chunks=chunks,
        chunk_embeddings=[[0.1] * 384, [0.2] * 384],
    )

    # Exact-token match ("QSE-5") that a pure embedding search could plausibly miss.
    hits = kg.retrieve_top_chunks_bm25("QSE-5 energy density", [doc_id], k=5)
    assert any("QSE-5" in content for _, content in hits)

    # A query restricted to a doc_id that doesn't exist yields nothing, not an error.
    assert kg.retrieve_top_chunks_bm25("QSE-5", [doc_id + 999], k=5) == []

    kg.db.close()
    del kg
    gc.collect()
    try:
        os.remove(db_path)
        os.rmdir(tmp_dir)
    except PermissionError:
        pass


def test_kg_bm25_match_expr_handles_special_characters():
    """Punctuation-heavy queries must not raise an FTS5 syntax error."""
    tmp_dir = tempfile.mkdtemp()
    db_path = os.path.join(tmp_dir, "test_kg_bm25_special.db")
    kg = KnowledgeGraph(db_path)

    doc_id = kg.store_document_and_chunks(
        url="http://test.com/x",
        query="q",
        summary="s",
        summary_embedding=[0.1] * 384,
        chunks=["Revenue grew 12% quarter-over-quarter (Q3 2026)."],
        chunk_embeddings=[[0.1] * 384],
    )

    # Hyphens, parens, %, colons — all would break a naive raw FTS5 MATCH string.
    hits = kg.retrieve_top_chunks_bm25('revenue: "growth" -- (Q3 2026)?!', [doc_id], k=5)
    assert isinstance(hits, list)  # must not raise

    kg.db.close()
    del kg
    gc.collect()
    try:
        os.remove(db_path)
        os.rmdir(tmp_dir)
    except PermissionError:
        pass


def test_kg_find_gaps_session_scoping():
    """find_gaps only surfaces weakly-supported claims from the requested session."""
    tmp_dir = tempfile.mkdtemp()
    db_path = os.path.join(tmp_dir, "test_kg_gaps.db")
    kg = KnowledgeGraph(db_path)

    kg.store_facts([_make_fact("Gap A", support="NOT_SUPPORTED", confidence=0.2)], session_id="sessA")
    kg.store_facts([_make_fact("Gap B", support="NOT_SUPPORTED", confidence=0.2)], session_id="sessB")

    emb = [0.1] * 384
    gaps_a = kg.find_gaps(emb, session_id="sessA")
    assert gaps_a == ["Gap A"]

    kg.db.close()
    del kg
    gc.collect()
    try:
        os.remove(db_path)
        os.rmdir(tmp_dir)
    except PermissionError:
        pass


# ── Evidence graph: sources, contradictions, consensus findings, gaps ────────

@pytest.fixture()
def kg():
    tmp_dir = tempfile.mkdtemp()
    db_path = os.path.join(tmp_dir, "test_kg_evidence.db")
    store = KnowledgeGraph(db_path)
    yield store
    store.db.close()
    del store
    gc.collect()
    try:
        os.remove(db_path)
        os.rmdir(tmp_dir)
    except PermissionError:
        pass


def test_store_and_read_sources(kg):
    kg.store_sources(
        [{"source_id": "[1]", "url": "http://a.com", "credibility_score": 0.9,
          "source_type": "Academic", "relevance_score": 0.8, "as_of_date": "2026",
          "snippet": "..."}],
        session_id="s1",
    )
    graph = kg.get_evidence_graph("s1")
    assert len(graph["sources"]) == 1
    assert graph["sources"][0]["url"] == "http://a.com"
    assert graph["sources"][0]["relevance_score"] == 0.8


def test_store_sources_is_idempotent_per_source_id(kg):
    """source_map accumulates across research-loop iterations; Verifier calls
    store_sources once per iteration, so re-storing the same source_id must
    not duplicate it."""
    source = {"source_id": "[1]", "url": "http://a.com", "credibility_score": 0.9}
    kg.store_sources([source], session_id="s1")
    kg.store_sources([source], session_id="s1")  # second pass, same source
    graph = kg.get_evidence_graph("s1")
    assert len(graph["sources"]) == 1


def test_store_sources_adds_only_new_ones_on_a_later_pass(kg):
    kg.store_sources([{"source_id": "[1]", "url": "http://a.com"}], session_id="s1")
    kg.store_sources(
        [{"source_id": "[1]", "url": "http://a.com"}, {"source_id": "[2]", "url": "http://b.com"}],
        session_id="s1",
    )
    graph = kg.get_evidence_graph("s1")
    assert {s["source_id"] for s in graph["sources"]} == {"[1]", "[2]"}


def test_sources_scoped_by_session(kg):
    kg.store_sources([{"source_id": "[1]", "url": "http://a.com"}], session_id="s1")
    kg.store_sources([{"source_id": "[1]", "url": "http://b.com"}], session_id="s2")
    assert len(kg.get_evidence_graph("s1")["sources"]) == 1
    assert kg.get_evidence_graph("s1")["sources"][0]["url"] == "http://a.com"


def test_store_and_read_contradictions(kg):
    kg.store_contradictions(
        [{"statement": "Conflicting valuations", "claims": ["A", "B"],
          "sources": ["http://a.com", "http://b.com"], "source_count": 2}],
        session_id="s1",
    )
    graph = kg.get_evidence_graph("s1")
    assert len(graph["contradictions"]) == 1
    entry = graph["contradictions"][0]
    assert entry["claims"] == ["A", "B"]  # JSON round-trips back to a list
    assert entry["source_count"] == 2


def test_store_and_read_consensus_findings(kg):
    kg.store_consensus_findings(
        [{"statement": "Agreed figure", "claims": ["A", "B"], "sources": ["u1"], "source_count": 2}],
        session_id="s1",
    )
    graph = kg.get_evidence_graph("s1")
    assert len(graph["consensus_findings"]) == 1
    assert graph["consensus_findings"][0]["statement"] == "Agreed figure"


def test_store_and_read_gaps(kg):
    kg.store_gaps(
        [{"gap_type": "coverage_gap", "description": "no data on X", "iteration": 0},
         {"gap_type": "knowledge_gap", "description": "claim Y unverified", "iteration": 1}],
        session_id="s1",
    )
    graph = kg.get_evidence_graph("s1")
    assert len(graph["gaps"]) == 2
    types = {g["gap_type"] for g in graph["gaps"]}
    assert types == {"coverage_gap", "knowledge_gap"}


def test_empty_store_calls_are_no_ops(kg):
    """None of the new store_* methods should touch the DB (or raise) on an
    empty list — Consensus/Critic call these even when nothing was found."""
    kg.store_sources([], session_id="s1")
    kg.store_contradictions([], session_id="s1")
    kg.store_consensus_findings([], session_id="s1")
    kg.store_gaps([], session_id="s1")
    graph = kg.get_evidence_graph("s1")
    assert graph == {"facts": [], "sources": [], "contradictions": [], "consensus_findings": [], "gaps": []}


def test_get_fact_detail_returns_none_for_unknown_fact(kg):
    assert kg.get_fact_detail("s1", 999) is None


def test_get_fact_detail_joins_source_and_contradictions(kg):
    kg.store_facts([_make_fact("SpaceX valued at $350B")], session_id="s1")
    kg.store_sources(
        [{"source_id": "[1]", "url": "http://test.com", "credibility_score": 0.9,
          "relevance_score": 0.7}],
        session_id="s1",
    )
    kg.store_contradictions(
        [{"statement": "conflict", "claims": ["SpaceX valued at $350B"],
          "sources": ["http://test.com"], "source_count": 1}],
        session_id="s1",
    )

    graph = kg.get_evidence_graph("s1")
    fact_id = graph["facts"][0]["id"]

    detail = kg.get_fact_detail("s1", fact_id)
    assert detail["claim"] == "SpaceX valued at $350B"
    assert detail["source"]["credibility_score"] == 0.9
    assert detail["source"]["relevance_score"] == 0.7
    assert len(detail["contradictions"]) == 1


def test_get_fact_detail_scoped_by_session():
    """A fact_id from another session must 404, not leak cross-session data —
    ids are globally auto-incrementing, session_id is the security boundary."""
    tmp_dir = tempfile.mkdtemp()
    db_path = os.path.join(tmp_dir, "test_kg_fact_scope.db")
    store = KnowledgeGraph(db_path)
    store.store_facts([_make_fact("secret to session A")], session_id="sessA")
    graph = store.get_evidence_graph("sessA")
    fact_id = graph["facts"][0]["id"]

    assert store.get_fact_detail("sessB", fact_id) is None

    store.db.close()
    del store
    gc.collect()
    try:
        os.remove(db_path)
        os.rmdir(tmp_dir)
    except PermissionError:
        pass
