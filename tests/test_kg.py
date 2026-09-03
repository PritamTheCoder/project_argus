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
