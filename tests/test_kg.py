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
