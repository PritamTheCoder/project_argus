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
