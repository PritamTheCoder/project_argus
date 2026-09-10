"""
Project Argus - Graph Orchestration Tests

Unit tests for the StateGraph wiring, conditional routing logic,
and persistence helpers. All tests are mocked — no API keys needed.
"""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock

from src.graph.builder import build_graph, route_after_critic, _critic_with_counter
from src.graph.persistence import generate_thread_id, get_run_config


# ── Graph Compilation Tests ─────────────────────────────────────────────────

@patch("src.graph.builder.librarian_node", return_value={})
@patch("src.graph.builder.scout_node", new_callable=AsyncMock, return_value={})
@patch("src.graph.builder.refiner_node", return_value={})
@patch("src.graph.builder.verifier_node", return_value={})
@patch("src.graph.builder.critic_node", return_value={"re_search_required": False, "critique": ""})
@patch("src.graph.builder.reflector_node", return_value={})
@patch("src.graph.builder.consensus_node", return_value={})
@patch("src.graph.builder.writer_node", return_value={})
def test_graph_compiles(mock_writer, mock_consensus, mock_reflector, mock_critic, mock_verifier, mock_refiner, mock_scout, mock_librarian):
    """build_graph() should return a compiled graph without errors."""
    graph = build_graph()
    assert graph is not None


@patch("src.graph.builder.librarian_node", return_value={})
@patch("src.graph.builder.scout_node", new_callable=AsyncMock, return_value={})
@patch("src.graph.builder.refiner_node", return_value={})
@patch("src.graph.builder.verifier_node", return_value={})
@patch("src.graph.builder.critic_node", return_value={"re_search_required": False, "critique": ""})
@patch("src.graph.builder.reflector_node", return_value={})
@patch("src.graph.builder.consensus_node", return_value={})
@patch("src.graph.builder.writer_node", return_value={})
def test_graph_node_names(mock_writer, mock_consensus, mock_reflector, mock_critic, mock_verifier, mock_refiner, mock_scout, mock_librarian):
    """Graph should contain all expected node names."""
    graph = build_graph()
    expected_nodes = {"librarian", "plan_gate", "scout", "refiner", "verifier", "fact_checker", "reflector", "consensus", "ghostwriter"}
    graph_nodes = set(graph.get_graph().nodes.keys()) - {"__start__", "__end__"}
    assert expected_nodes == graph_nodes


# ── Routing Logic Tests ─────────────────────────────────────────────────────

def test_route_to_consensus_when_no_re_search():
    """When re_search_required is False, should route to consensus (then writer)."""
    state = {"re_search_required": False, "iteration_count": 0}
    assert route_after_critic(state) == "consensus"


def test_route_to_scout_when_re_search_no_gaps():
    """When re_search is True, under loop cap, and no specific gaps: route to scout."""
    state = {
        "re_search_required": True,
        "iteration_count": 0,
        "knowledge_gap_detected": False,
        "knowledge_gaps": [],
    }
    assert route_after_critic(state) == "scout"

    state["iteration_count"] = 1
    assert route_after_critic(state) == "scout"


def test_route_to_reflector_when_re_search_with_gaps():
    """When re_search is True, under loop cap, and specific gaps exist: route to reflector."""
    state = {
        "re_search_required": True,
        "iteration_count": 0,
        "knowledge_gap_detected": True,
        "knowledge_gaps": ["missing data about X"],
    }
    assert route_after_critic(state) == "reflector"


@patch("src.graph.builder.MAX_RESEARCH_LOOPS", 2)
def test_route_to_consensus_at_max_loops():
    """When iteration_count >= MAX, should route to consensus even if re_search is True."""
    state = {"re_search_required": True, "iteration_count": 2}
    assert route_after_critic(state) == "consensus"

    state = {"re_search_required": True, "iteration_count": 5}
    assert route_after_critic(state) == "consensus"


def test_route_defaults_to_consensus_with_missing_keys():
    """Should default to consensus when state keys are missing."""
    state = {}
    assert route_after_critic(state) == "consensus"


def test_route_to_scout_when_credibility_low_even_with_gaps():
    """Low avg source credibility should override gap-chasing: broaden the
    search instead of sending the Reflector after individual claims that tend
    to just re-find the same low-credibility source."""
    state = {
        "re_search_required": True,
        "iteration_count": 0,
        "knowledge_gap_detected": True,
        "knowledge_gaps": ["some claim"],
        "quality_score": {"avg_source_credibility": 0.3},
    }
    assert route_after_critic(state) == "scout"


def test_route_to_reflector_when_credibility_fine():
    """Above the threshold, gap-chasing behaves as before."""
    state = {
        "re_search_required": True,
        "iteration_count": 0,
        "knowledge_gap_detected": True,
        "knowledge_gaps": ["some claim"],
        "quality_score": {"avg_source_credibility": 0.9},
    }
    assert route_after_critic(state) == "reflector"


# ── Critic Counter Wrapper Test ─────────────────────────────────────────────

@patch("src.graph.builder.critic_node")
def test_critic_with_counter_increments(mock_critic):
    """_critic_with_counter should call critic_node and bump iteration_count."""
    mock_critic.return_value = {"critique": "ok", "re_search_required": False}

    state = {"query": "test", "iteration_count": 0, "structured_evidence": []}
    result = _critic_with_counter(state)

    assert result["iteration_count"] == 1
    assert result["critique"] == "ok"
    mock_critic.assert_called_once_with(state)


@patch("src.graph.builder.critic_node")
def test_critic_with_counter_handles_missing_count(mock_critic):
    """Should default to 0 if iteration_count is missing from state."""
    mock_critic.return_value = {"critique": "good", "re_search_required": False}

    state = {"query": "test", "structured_evidence": []}
    result = _critic_with_counter(state)

    assert result["iteration_count"] == 1


# ── Persistence Helper Tests ───────────────────────────────────────────────

def test_generate_thread_id_is_unique():
    """Should return unique thread IDs."""
    id1 = generate_thread_id()
    id2 = generate_thread_id()
    assert id1 != id2
    assert isinstance(id1, str)
    assert len(id1) == 36  # UUID4 format


def test_get_run_config_structure():
    """Config dict should have the correct structure."""
    config = get_run_config("test-thread-123")
    assert config == {"configurable": {"thread_id": "test-thread-123"}}


def test_get_run_config_preserves_thread_id():
    """Thread ID should be preserved exactly as passed."""
    tid = generate_thread_id()
    config = get_run_config(tid)
    assert config["configurable"]["thread_id"] == tid


# ── fork_thread / list_checkpoints ──────────────────────────────────────────
# Real graph run against a real SQLite checkpointer (nodes mocked), so the
# as_node fix is proven against actual LangGraph behavior, not a mental model.

import tempfile
import pytest
from src.graph.persistence import fork_thread, list_checkpoints


@pytest.fixture()
def two_pass_graph():
    """A graph whose mocked nodes loop back to scout exactly once, so a
    genuine mid-loop checkpoint exists to fork from."""
    from src.graph import builder as builder_mod

    passes = {"n": 0}

    def librarian(state):
        return {"plan": [{"query": "q1", "mode": "MIXED"}], "original_plan": [{"query": "q1", "mode": "MIXED"}]}

    def scout(state):
        return {"scraped_data": [{"source_id": "[1]", "url": "http://a.com", "content": "x", "query": "q1"}]}

    def refiner(state):
        return {"structured_evidence": [{"claim": "c1", "source_id": "[1]", "source_excerpt": "x"}]}

    def verifier(state):
        return {"verified_facts": [], "knowledge_gap_detected": False, "knowledge_gaps": []}

    def critic(state):
        passes["n"] += 1
        if passes["n"] == 1:
            return {"re_search_required": True, "critique": "need more", "plan": [{"query": "q2", "mode": "MIXED"}]}
        return {"re_search_required": False, "critique": "sufficient"}

    def reflector(state):
        return {}

    def consensus(state):
        return {"consensus_findings": [], "contradictions": [], "quality_score": {}}

    def writer(state):
        return {"report": "FINAL REPORT"}

    patches = [
        patch.object(builder_mod, "librarian_node", librarian),
        patch.object(builder_mod, "scout_node", scout),
        patch.object(builder_mod, "refiner_node", refiner),
        patch.object(builder_mod, "verifier_node", verifier),
        patch.object(builder_mod, "critic_node", critic),
        patch.object(builder_mod, "reflector_node", reflector),
        patch.object(builder_mod, "consensus_node", consensus),
        patch.object(builder_mod, "writer_node", writer),
    ]
    for p in patches:
        p.start()
    yield builder_mod
    for p in patches:
        p.stop()


@pytest.mark.asyncio
async def test_fork_thread_resumes_at_the_right_node_not_the_start(two_pass_graph):
    """The core regression: forking a mid-loop checkpoint must resume at that
    checkpoint's node, not silently restart from librarian."""
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
    from src.graph.persistence import generate_thread_id, get_run_config

    tmp_dir = tempfile.mkdtemp()
    db_path = f"{tmp_dir}/checkpoints.db"
    seed = {
        "session_id": "", "query": "q", "plan": [], "original_plan": [],
        "scraped_data": [], "structured_evidence": [], "source_map": {},
        "backend_health": {}, "critique": "", "report": "", "re_search_required": False,
        "knowledge_gap_detected": False, "knowledge_gaps": [], "coverage_gaps": [],
        "gap_queries": [], "verified_facts": [], "iteration_count": 0, "active_node": "",
    }

    async with AsyncSqliteSaver.from_conn_string(db_path) as checkpointer:
        graph = two_pass_graph.build_graph(checkpointer=checkpointer)

        thread_a = generate_thread_id()
        config_a = get_run_config(thread_a)
        seed_a = {**seed, "session_id": thread_a}
        async for _ in graph.astream(seed_a, config=config_a):
            pass

        history = await list_checkpoints(graph, thread_a)
        # The checkpoint right after the first critic pass, before the second
        # scout run — the point a "branch" action would fork from.
        loop_back = next(
            c for c in history if c["next"] == ["scout"] and c["iteration_count"] == 1
        )

        new_thread_id = await fork_thread(
            graph, thread_a,
            {"plan": [{"query": "q2", "mode": "MIXED"}, {"query": "BRANCH", "mode": "MIXED"}]},
            checkpoint_id=loop_back["checkpoint_id"],
        )
        assert new_thread_id != thread_a

        config_b = get_run_config(new_thread_id)
        seeded = await graph.aget_state(config_b)
        assert seeded.next == ("scout",), "must resume at scout, not restart at librarian"

        call_order = []
        orig_scout = two_pass_graph.scout_node
        with patch.object(two_pass_graph, "scout_node", lambda s: (call_order.append("scout"), orig_scout(s))[1]):
            async for event in graph.astream(None, config=config_b):
                call_order.extend(event.keys())

        assert "librarian" not in call_order, "resuming must not re-run the entry node"

        final_b = await graph.aget_state(config_b)
        assert final_b.values["report"] == "FINAL REPORT"
        assert any(q["query"] == "BRANCH" for q in final_b.values["plan"]), \
            "the injected direction must survive to completion"

        # The source thread must be provably untouched by the fork.
        final_a = await graph.aget_state(config_a)
        assert final_a.values["report"] == "FINAL REPORT"
        assert not any("BRANCH" in q.get("query", "") for q in final_a.values.get("plan", []))


@pytest.mark.asyncio
async def test_naive_update_state_without_as_node_resets_to_the_start(two_pass_graph):
    """The failure mode fork_thread's as_node exists to avoid. Must keep
    failing if LangGraph's behavior here ever changes."""
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
    from src.graph.persistence import generate_thread_id, get_run_config

    tmp_dir = tempfile.mkdtemp()
    db_path = f"{tmp_dir}/checkpoints_naive.db"
    seed = {
        "session_id": "", "query": "q", "plan": [], "original_plan": [],
        "scraped_data": [], "structured_evidence": [], "source_map": {},
        "backend_health": {}, "critique": "", "report": "", "re_search_required": False,
        "knowledge_gap_detected": False, "knowledge_gaps": [], "coverage_gaps": [],
        "gap_queries": [], "verified_facts": [], "iteration_count": 0, "active_node": "",
    }

    async with AsyncSqliteSaver.from_conn_string(db_path) as checkpointer:
        graph = two_pass_graph.build_graph(checkpointer=checkpointer)
        thread_a = generate_thread_id()
        config_a = get_run_config(thread_a)
        async for _ in graph.astream({**seed, "session_id": thread_a}, config=config_a):
            pass

        history = await list_checkpoints(graph, thread_a)
        loop_back = next(c for c in history if c["next"] == ["scout"] and c["iteration_count"] == 1)
        source_config = {"configurable": {"thread_id": thread_a, "checkpoint_id": loop_back["checkpoint_id"]}}
        snapshot = await graph.aget_state(source_config)

        thread_b = generate_thread_id()
        config_b = get_run_config(thread_b)
        await graph.aupdate_state(config_b, dict(snapshot.values))  # no as_node

        naive = await graph.aget_state(config_b)
        assert naive.next == ("librarian",), (
            "if this ever changes, fork_thread's as_node workaround may no "
            "longer be necessary — re-check before removing it"
        )


@pytest.mark.asyncio
async def test_fork_thread_raises_for_unknown_checkpoint(two_pass_graph):
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
    from src.graph.persistence import generate_thread_id, get_run_config

    tmp_dir = tempfile.mkdtemp()
    db_path = f"{tmp_dir}/checkpoints_missing.db"
    async with AsyncSqliteSaver.from_conn_string(db_path) as checkpointer:
        graph = two_pass_graph.build_graph(checkpointer=checkpointer)
        thread_a = generate_thread_id()
        with pytest.raises(ValueError):
            await fork_thread(graph, thread_a, {}, checkpoint_id="nonexistent-checkpoint-id")


# ── dig-deeper: forcing re-entry at scout from a finished run ───────────────
# Unlike branch (which preserves a checkpoint's natural .next), dig-deeper
# forks from a run that already reached END (.next == ()) and must force its
# way back to "scout" regardless. as_node="reflector" does this because
# reflector's only outgoing edge is a plain, unconditional one to scout.

@pytest.mark.asyncio
async def test_dig_deeper_forces_scout_reentry_from_a_finished_run(two_pass_graph):
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
    from src.graph.persistence import generate_thread_id, get_run_config

    tmp_dir = tempfile.mkdtemp()
    db_path = f"{tmp_dir}/checkpoints_digdeeper.db"
    seed = {
        "session_id": "", "query": "q", "plan": [], "original_plan": [],
        "scraped_data": [], "structured_evidence": [], "source_map": {},
        "backend_health": {}, "critique": "", "report": "", "re_search_required": False,
        "knowledge_gap_detected": False, "knowledge_gaps": [], "coverage_gaps": [],
        "gap_queries": [], "verified_facts": [], "iteration_count": 0, "active_node": "",
    }

    async with AsyncSqliteSaver.from_conn_string(db_path) as checkpointer:
        graph = two_pass_graph.build_graph(checkpointer=checkpointer)
        thread_a = generate_thread_id()
        config_a = get_run_config(thread_a)
        async for _ in graph.astream({**seed, "session_id": thread_a}, config=config_a):
            pass

        finished = await graph.aget_state(config_a)
        assert finished.next == (), "the source run must have actually finished"

        new_thread_id = await fork_thread(
            graph, thread_a,
            {"plan": [{"query": "DIG DEEPER: extra detail on c1", "mode": "MIXED"}]},
            as_node="reflector",
        )
        config_b = get_run_config(new_thread_id)
        seeded = await graph.aget_state(config_b)
        assert seeded.next == ("scout",), "must re-enter at scout even though the source run had already ended"

        async for _ in graph.astream(None, config=config_b):
            pass

        final_b = await graph.aget_state(config_b)
        assert final_b.values["report"] == "FINAL REPORT"
        assert any(
            q["query"] == "DIG DEEPER: extra detail on c1" for q in final_b.values["plan"]
        )

        # The source run's own final state must be unaffected.
        finished_recheck = await graph.aget_state(config_a)
        assert finished_recheck.values == finished.values
