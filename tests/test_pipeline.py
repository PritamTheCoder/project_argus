"""
Project Argus - Pipeline tests: plan approval gate (interrupt/resume)

Real build_graph() and a real SQLite checkpointer, mocked agent bodies, so
the interrupt()/Command(resume=...) mechanics are proven against actual
LangGraph behavior — not asserted against a mental model of it.
"""

from unittest.mock import patch

import pytest

from src.graph import builder as builder_mod
from src.graph.pipeline import astream_research


def _mock_nodes():
    def librarian(state):
        return {"plan": [{"query": "q1", "mode": "MIXED"}], "original_plan": [{"query": "q1", "mode": "MIXED"}]}

    def scout(state):
        return {"scraped_data": []}

    def refiner(state):
        return {"structured_evidence": []}

    def verifier(state):
        return {"verified_facts": [], "knowledge_gap_detected": False, "knowledge_gaps": []}

    def critic(state):
        return {"re_search_required": False, "critique": "sufficient"}

    def reflector(state):
        return {}

    def consensus(state):
        return {"consensus_findings": [], "contradictions": [], "quality_score": {}}

    def writer(state):
        return {"report": "FINAL REPORT, plan=" + str(state.get("plan"))}

    return [
        patch.object(builder_mod, "librarian_node", librarian),
        patch.object(builder_mod, "scout_node", scout),
        patch.object(builder_mod, "refiner_node", refiner),
        patch.object(builder_mod, "verifier_node", verifier),
        patch.object(builder_mod, "critic_node", critic),
        patch.object(builder_mod, "reflector_node", reflector),
        patch.object(builder_mod, "consensus_node", consensus),
        patch.object(builder_mod, "writer_node", writer),
    ]


@pytest.fixture()
def mocked_graph():
    patches = _mock_nodes()
    for p in patches:
        p.start()
    yield
    for p in patches:
        p.stop()


@pytest.fixture()
def db_path(tmp_path):
    return str(tmp_path / "checkpoints.db")


@pytest.mark.asyncio
async def test_default_run_never_pauses(mocked_graph, db_path):
    """require_approval defaults False — plan_gate must be a true no-op."""
    with patch("src.graph.persistence.DB_PATH", db_path):
        events = [
            (name, update)
            async for name, update in astream_research("q", thread_id="t1")
        ]
    names = [n for n, _ in events]
    assert "__interrupt__" not in names
    assert names[-1] == "__final__"


@pytest.mark.asyncio
async def test_require_approval_pauses_before_scout(mocked_graph, db_path):
    with patch("src.graph.persistence.DB_PATH", db_path):
        events = [
            (name, update)
            async for name, update in astream_research("q", thread_id="t2", require_approval=True)
        ]
    names = [n for n, _ in events]
    assert names[-1] == "__interrupt__"
    assert "__final__" not in names
    # scout never ran — the whole point of pausing before it.
    assert "scout" not in names

    pending = dict(events)["__interrupt__"]
    assert pending["pending_plan"] == [{"query": "q1", "mode": "MIXED"}]


@pytest.mark.asyncio
async def test_resume_with_edited_plan_completes_and_uses_it(mocked_graph, db_path):
    with patch("src.graph.persistence.DB_PATH", db_path):
        async for _ in astream_research("q", thread_id="t3", require_approval=True):
            pass

        edited = [{"query": "q1", "mode": "MIXED"}, {"query": "HUMAN ADDED", "mode": "MIXED"}]
        events = [
            (name, update)
            async for name, update in astream_research(thread_id="t3", resume_value=edited)
        ]

    names = [n for n, _ in events]
    assert names[-1] == "__final__"
    final = dict(events)["__final__"]
    assert "HUMAN ADDED" in final["report"]


@pytest.mark.asyncio
async def test_resume_does_not_rerun_librarian(mocked_graph, db_path):
    """plan_gate's own pre-interrupt line re-executes on resume (a LangGraph
    guarantee, not a bug) — but nodes BEFORE plan_gate must not re-run."""
    calls = {"librarian": 0}

    def counting_librarian(state):
        calls["librarian"] += 1
        return {"plan": [{"query": "q1", "mode": "MIXED"}], "original_plan": [{"query": "q1", "mode": "MIXED"}]}

    with patch.object(builder_mod, "librarian_node", counting_librarian):
        with patch("src.graph.persistence.DB_PATH", db_path):
            async for _ in astream_research("q", thread_id="t4", require_approval=True):
                pass
            assert calls["librarian"] == 1

            async for _ in astream_research(thread_id="t4", resume_value=[{"query": "q1", "mode": "MIXED"}]):
                pass
            assert calls["librarian"] == 1, "librarian must not re-run on resume"
