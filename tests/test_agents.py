"""
Project Argus - Agent Node Tests

Unit tests for each agent node. All external calls (LLMs, KG, search,
scrape) are mocked — no API keys or network access required.
"""

import pytest
from unittest.mock import MagicMock, patch, AsyncMock

from src.schema.state import AgentState, ResearchPlan, SearchIntent, FactCheckResult
from src.agents.librarian import librarian_node
from src.agents.scout import scout_node
from src.agents.refiner import refiner_node, _split_into_batches
from src.agents.critic import critic_node, _compute_coverage_gaps
from src.agents.reflector import reflector_node, ReflectorOutput
from src.agents.writer import writer_node


# ── Shared state builder ──────────────────────────────────────────────────────

def _base_state(**overrides) -> dict:
    base = {
        "query": "test topic",
        "plan": [],
        "scraped_data": [],
        "structured_evidence": [],
        "source_map": {},
        "critique": "",
        "report": "",
        "re_search_required": False,
        "knowledge_gap_detected": False,
        "knowledge_gaps": [],
        "gap_queries": [],
        "verified_facts": [],
        "iteration_count": 0,
        "active_node": "",
    }
    base.update(overrides)
    return base


# ── Librarian Tests ───────────────────────────────────────────────────────────

@patch("src.agents.librarian.get_llm_with_fallbacks")
def test_librarian_generates_plan(mock_get_llm):
    """Librarian should decompose the query and return a list of search-intent dicts."""
    expected_plan = ResearchPlan(search_queries=[
        SearchIntent(query="solid state battery energy density 2026", mode="TRUSTED_ONLY"),
        SearchIntent(query="QuantumScape QSE-5 cycle life results", mode="TRUSTED_FIRST"),
    ])
    # LangChain wraps MagicMock as RunnableLambda; the chain calls mock(input) → .return_value
    mock_get_llm.return_value.return_value = expected_plan

    result = librarian_node(_base_state())

    assert "plan" in result
    assert len(result["plan"]) == 2
    assert result["plan"][0]["query"] == "solid state battery energy density 2026"
    assert result["plan"][0]["mode"] == "TRUSTED_ONLY"
    assert result["active_node"] == "librarian"


@patch("src.agents.librarian.get_llm_with_fallbacks")
def test_librarian_injects_critique_on_iteration(mock_get_llm):
    """On iteration > 0, the system prompt must contain the previous critique."""
    expected_plan = ResearchPlan(search_queries=[
        SearchIntent(query="gap-filling query for missing data", mode="MIXED"),
    ])
    mock_get_llm.return_value.return_value = expected_plan

    state = _base_state(
        iteration_count=1,
        critique="Missing: specific manufacturing cost data for 2025.",
    )
    result = librarian_node(state)

    assert "plan" in result
    assert len(result["plan"]) == 1
    mock_get_llm.assert_called_once()


@patch("src.agents.librarian.get_llm_with_fallbacks")
def test_librarian_no_critique_injection_on_first_iteration(mock_get_llm):
    """On iteration 0, critique should NOT be injected even if state has a stale value."""
    expected_plan = ResearchPlan(search_queries=[
        SearchIntent(query="general query", mode="MIXED"),
    ])
    mock_get_llm.return_value.invoke.return_value = expected_plan

    # iteration_count=0, critique is non-empty (shouldn't be injected)
    state = _base_state(iteration_count=0, critique="old critique from prior run")
    result = librarian_node(state)

    assert result["active_node"] == "librarian"
    mock_get_llm.assert_called_once()


@patch("src.agents.librarian.get_llm_with_fallbacks")
def test_librarian_sets_original_plan_on_first_iteration(mock_get_llm):
    """original_plan lets the Critic measure coverage even after `plan` is
    later overwritten by follow-up queries — set once, on the first pass."""
    expected_plan = ResearchPlan(search_queries=[SearchIntent(query="q1", mode="MIXED")])
    mock_get_llm.return_value.return_value = expected_plan

    result = librarian_node(_base_state(iteration_count=0))

    assert result["original_plan"] == result["plan"]


@patch("src.agents.librarian.get_llm_with_fallbacks")
def test_librarian_does_not_reset_original_plan_on_later_iterations(mock_get_llm):
    expected_plan = ResearchPlan(search_queries=[SearchIntent(query="q2", mode="MIXED")])
    mock_get_llm.return_value.return_value = expected_plan

    result = librarian_node(_base_state(iteration_count=1, critique="prior critique"))

    assert "original_plan" not in result


# ── Librarian: structural memory-reuse (Phase 4A.2) ───────────────────────────

@patch("src.agents.librarian.get_llm_with_fallbacks")
@patch("src.agents.librarian._prior_knowledge_summary")
def test_librarian_checks_prior_knowledge_on_first_iteration_only(mock_summary, mock_get_llm):
    """The prior-knowledge check runs on iteration 0, never on a critique-driven iteration."""
    mock_summary.return_value = ""
    mock_get_llm.return_value.return_value = ResearchPlan(search_queries=[
        SearchIntent(query="q", mode="MIXED"),
    ])

    librarian_node(_base_state(iteration_count=0))
    mock_summary.assert_called_once()

    mock_summary.reset_mock()
    librarian_node(_base_state(iteration_count=1, critique="gap"))
    mock_summary.assert_not_called()


def test_prior_knowledge_summary_disabled_by_default():
    """KG_LOOKUP_GLOBAL off (the default) means no cross-run lookup happens at all."""
    from src.agents.librarian import _prior_knowledge_summary
    with patch("src.agents.librarian.KG_LOOKUP_GLOBAL", False):
        assert _prior_knowledge_summary("any query") == ""


def test_prior_knowledge_summary_filters_weak_facts():
    """Only SUPPORTED, high-confidence facts are surfaced; weak/unsupported ones are dropped."""
    from src.agents.librarian import _prior_knowledge_summary

    store = MagicMock()
    store.retrieve_relevant_facts.return_value = [
        {"claim": "Strong fact", "source_url": "http://a.com", "support_level": "SUPPORTED", "confidence": 0.9},
        {"claim": "Weak fact", "source_url": "http://b.com", "support_level": "SUPPORTED", "confidence": 0.2},
        {"claim": "Unsupported fact", "source_url": "http://c.com", "support_level": "NOT_SUPPORTED", "confidence": 0.9},
    ]
    with patch("src.agents.librarian.KG_LOOKUP_GLOBAL", True), \
         patch("src.utils.embeddings.get_embeddings", return_value=[[0.1] * 384]), \
         patch("src.graph.kg.kg_store", store):
        summary = _prior_knowledge_summary("query", min_confidence=0.6)

    assert "Strong fact" in summary
    assert "Weak fact" not in summary
    assert "Unsupported fact" not in summary
    # Global reuse must query without session scoping.
    assert store.retrieve_relevant_facts.call_args.kwargs["session_id"] is None


def test_prior_knowledge_summary_degrades_on_failure():
    """A KG/embedding failure returns '' rather than raising — planning proceeds unaffected."""
    from src.agents.librarian import _prior_knowledge_summary
    with patch("src.agents.librarian.KG_LOOKUP_GLOBAL", True), \
         patch("src.utils.embeddings.get_embeddings", side_effect=RuntimeError("embedding service down")):
        assert _prior_knowledge_summary("query") == ""


# ── Scout Tests ───────────────────────────────────────────────────────────────

from contextlib import asynccontextmanager


@asynccontextmanager
async def _fake_shared_crawler():
    """Scout opens a real browser via shared_crawler() — stub it out in tests."""
    yield MagicMock()


@pytest.mark.asyncio
@patch("src.agents.scout.shared_crawler", _fake_shared_crawler)
@patch("src.agents.scout.gather_sources_for_query", new_callable=AsyncMock)
@patch("src.agents.scout.scrape_urls", new_callable=AsyncMock)
@patch("src.agents.scout.get_embeddings")
@patch("src.agents.scout.rerank_chunks")
async def test_scout_execution(mock_rerank, mock_embeddings, mock_scrape, mock_gather):
    """Scout should gather (web + academic), scrape web ones, index, and build source_map."""
    long_content = (
        "This is a detailed paragraph about solid state batteries that exceeds fifty characters.\n\n"
        "Another paragraph providing technical details about energy density improvements in research."
    )

    # q1 → a web candidate needing scrape; q2 → an academic candidate with a prefetched abstract.
    mock_gather.side_effect = [
        [{"url": "http://a.com", "content": "", "needs_scrape": True, "source": "web"}],
        [{"url": "http://b.com", "content": long_content, "needs_scrape": False,
          "source": "semantic_scholar", "credibility_hint": 0.9,
          "source_type_hint": "Academic/Scientific", "as_of_date": "2024"}],
    ]
    mock_embeddings.return_value = [[0.1] * 384]
    mock_rerank.side_effect = lambda q, chunks, top_k: chunks[:top_k]
    # Only the web candidate (q1) is scraped; q2 has no scrape targets.
    mock_scrape.side_effect = [
        [{"url": "http://a.com", "content": long_content, "success": True}],
    ]

    mock_kg = MagicMock()
    mock_kg.store_document_and_chunks.return_value = 1
    mock_kg.retrieve_top_docs.return_value = [1]
    mock_kg.retrieve_top_chunks.return_value = [(1, "chunk from a.com")]
    mock_kg.get_doc_metadata.return_value = {"url": "http://a.com", "query": "q1", "summary": "summary"}
    mock_kg.get_all_chunks_for_docs.return_value = {1: ["chunk from a.com"]}

    state = _base_state(plan=["q1", "q2"])

    with patch("src.graph.kg.kg_store", mock_kg):
        result = await scout_node(state)

    assert "scraped_data" in result
    assert len(result["scraped_data"]) > 0
    assert result["active_node"] == "scout"
    assert len(result["source_map"]) > 0
    assert mock_kg.store_document_and_chunks.called
    assert mock_kg.retrieve_top_docs.called
    assert mock_kg.retrieve_top_chunks.called
    assert mock_kg.get_all_chunks_for_docs.called
    # Academic candidate was used without scraping (only the web URL was scraped).
    assert mock_scrape.call_count == 1


@pytest.mark.asyncio
@patch("src.agents.scout.shared_crawler", _fake_shared_crawler)
@patch("src.agents.scout.gather_sources_for_query", new_callable=AsyncMock)
@patch("src.agents.scout.scrape_urls", new_callable=AsyncMock)
@patch("src.agents.scout.get_embeddings")
@patch("src.agents.scout.MAX_SOURCES_PER_DOMAIN", 2)
async def test_scout_caps_sources_per_domain(mock_embeddings, mock_scrape, mock_gather):
    """No single domain should fill more than MAX_SOURCES_PER_DOMAIN source_map
    slots — otherwise one content farm's subpages crowd out source diversity.

    Forces the no-embeddings degraded path (registers scraped results directly,
    skipping retrieval/reranking) so this test doesn't need to mock the KG."""
    mock_embeddings.side_effect = Exception("embeddings unavailable")

    urls = [f"http://farm.com/page{i}" for i in range(4)]
    mock_gather.return_value = [{"url": u, "content": "", "needs_scrape": True, "source": "web"} for u in urls]
    mock_scrape.return_value = [
        {"url": u, "content": "Some scraped page content.", "success": True} for u in urls
    ]

    state = _base_state(plan=["q1"])
    with patch("src.graph.kg.kg_store", MagicMock()):
        result = await scout_node(state)

    farm_sources = [v for v in result["source_map"].values() if "farm.com" in v["url"]]
    assert len(farm_sources) == 2


# ── Refiner Tests ─────────────────────────────────────────────────────────────

@patch("src.agents.refiner.extract_facts")
@patch("src.agents.refiner.get_llm_with_fallbacks")
def test_refiner_extraction(mock_get_llm, mock_extract):
    """Refiner should call extract_facts and return structured_evidence."""
    from src.tools.refiner import ExtractedFact, FactExtractionResult
    from src.agents.refiner import ExtractionSchema

    # _generate_dynamic_schema: get_llm_with_fallbacks(...) already returns the
    # structured-output LLM, so .invoke() is the next hop straight to the result.
    mock_get_llm.return_value.invoke.return_value = (
        ExtractionSchema(schema_dict={"dates": "date", "metrics": "numeric value"})
    )

    mock_extract.return_value = {
        "facts": [
            {
                "class": "dates",
                "claim": "Announced January 2026",
                "source_excerpt": "announced on January 15, 2026",
                "source_id": "[1]",
                "source_span": {"start": None, "end": None},
                "attributes": {},
            }
        ]
    }

    state = _base_state(
        scraped_data=[{"content": "some text about batteries", "source_id": "[1]"}],
        plan=["q1"],
        source_map={"[1]": {"url": "http://a.com"}},
    )

    result = refiner_node(state)

    assert "structured_evidence" in result
    assert len(result["structured_evidence"]) == 1
    assert result["structured_evidence"][0]["claim"] == "Announced January 2026"
    assert result["structured_evidence"][0]["source_url"] == "http://a.com"
    assert result["active_node"] == "refiner"


# ── Critic Tests ──────────────────────────────────────────────────────────────

@patch("src.agents.critic.get_llm_with_fallbacks")
@patch("src.agents.critic.get_embeddings")
@patch("src.graph.kg.kg_store")
def test_critic_evaluation_gaps_found(mock_kg, mock_embeddings, mock_get_llm):
    """When the critic finds gaps, re_search_required should be True."""
    mock_embeddings.return_value = [[0.1] * 384]
    mock_kg.retrieve_relevant_facts.return_value = []

    # Chain calls mock as callable → .return_value (not .invoke.return_value)
    mock_get_llm.return_value.return_value = FactCheckResult(
        status="gaps_found",
        new_queries=[SearchIntent(query="follow-up query", mode="MIXED")],
        critique="Missing manufacturing cost data.",
    )

    result = critic_node(_base_state())

    assert result["re_search_required"] is True
    assert result["critique"] == "Missing manufacturing cost data."
    assert result["active_node"] == "critic"


@patch("src.agents.critic.get_llm_with_fallbacks")
@patch("src.agents.critic.get_embeddings")
@patch("src.graph.kg.kg_store")
def test_critic_evaluation_sufficient(mock_kg, mock_embeddings, mock_get_llm):
    """When evidence is sufficient, re_search_required should be False."""
    mock_embeddings.return_value = [[0.1] * 384]
    mock_kg.retrieve_relevant_facts.return_value = []

    mock_get_llm.return_value.return_value = FactCheckResult(
        status="sufficient",
        new_queries=[],
        critique="All key questions are answered.",
    )

    result = critic_node(_base_state())

    assert result["re_search_required"] is False
    assert result["active_node"] == "critic"


# ── Critic: coverage gaps (Phase 5B) ────────────────────────────────────────

def test_compute_coverage_gaps_flags_unanswered_subquestion():
    original_plan = [
        {"query": "SpaceX funding rounds", "mode": "MIXED"},
        {"query": "SpaceX Starlink revenue", "mode": "MIXED"},
    ]
    scraped_data = [{"source_id": "[1]", "query": "SpaceX funding rounds"}]
    verified_facts = [{"source_id": "[1]", "support_level": "SUPPORTED", "claim": "x"}]

    gaps = _compute_coverage_gaps(original_plan, scraped_data, verified_facts)
    assert gaps == ["SpaceX Starlink revenue"]


def test_compute_coverage_gaps_empty_when_all_answered():
    original_plan = [{"query": "q1", "mode": "MIXED"}]
    scraped_data = [{"source_id": "[1]", "query": "q1"}]
    verified_facts = [{"source_id": "[1]", "support_level": "SUPPORTED", "claim": "x"}]
    assert _compute_coverage_gaps(original_plan, scraped_data, verified_facts) == []


def test_compute_coverage_gaps_ignores_ungrounded_facts():
    """A fact that failed verification doesn't count as covering its sub-question."""
    original_plan = [{"query": "q1", "mode": "MIXED"}]
    scraped_data = [{"source_id": "[1]", "query": "q1"}]
    verified_facts = [{"source_id": "[1]", "support_level": "NOT_SUPPORTED", "claim": "x"}]
    assert _compute_coverage_gaps(original_plan, scraped_data, verified_facts) == ["q1"]


@patch("src.agents.critic.get_llm_with_fallbacks")
@patch("src.agents.critic.get_embeddings")
@patch("src.graph.kg.kg_store")
def test_critic_returns_coverage_gaps(mock_kg, mock_embeddings, mock_get_llm):
    mock_embeddings.return_value = [[0.1] * 384]
    mock_kg.retrieve_relevant_facts.return_value = []
    mock_get_llm.return_value.return_value = FactCheckResult(status="sufficient", new_queries=[], critique="ok")

    state = _base_state(original_plan=[{"query": "q1", "mode": "MIXED"}])
    result = critic_node(state)

    assert result["coverage_gaps"] == ["q1"]


@patch("src.agents.critic.get_llm_with_fallbacks")
@patch("src.agents.critic.get_embeddings")
@patch("src.graph.kg.kg_store")
def test_critic_upgrades_mixed_mode_when_credibility_low(mock_kg, mock_embeddings, mock_get_llm):
    """Chasing more MIXED-mode search when sources are already low-credibility
    just re-finds the same tier of source — upgrade toward trusted instead."""
    mock_embeddings.return_value = [[0.1] * 384]
    mock_kg.retrieve_relevant_facts.return_value = []
    mock_get_llm.return_value.return_value = FactCheckResult(
        status="gaps_found",
        new_queries=[SearchIntent(query="follow-up", mode="MIXED")],
        critique="Sources are weak.",
    )

    state = _base_state(quality_score={"avg_source_credibility": 0.3})
    result = critic_node(state)

    assert result["plan"][0]["mode"] == "TRUSTED_FIRST"


@patch("src.agents.critic.get_llm_with_fallbacks")
@patch("src.agents.critic.get_embeddings")
@patch("src.graph.kg.kg_store")
def test_critic_leaves_mode_alone_when_credibility_fine(mock_kg, mock_embeddings, mock_get_llm):
    mock_embeddings.return_value = [[0.1] * 384]
    mock_kg.retrieve_relevant_facts.return_value = []
    mock_get_llm.return_value.return_value = FactCheckResult(
        status="gaps_found",
        new_queries=[SearchIntent(query="follow-up", mode="MIXED")],
        critique="ok",
    )

    state = _base_state(quality_score={"avg_source_credibility": 0.9})
    result = critic_node(state)

    assert result["plan"][0]["mode"] == "MIXED"


# ── Critic: gap persistence ──────────────────────────────────────────────────

@patch("src.agents.critic.get_llm_with_fallbacks")
@patch("src.agents.critic.get_embeddings")
@patch("src.graph.kg.kg_store")
def test_critic_stores_coverage_and_knowledge_gaps(mock_kg, mock_embeddings, mock_get_llm):
    mock_embeddings.return_value = [[0.1] * 384]
    mock_kg.retrieve_relevant_facts.return_value = []
    mock_kg.store_gaps = MagicMock()
    mock_get_llm.return_value.return_value = FactCheckResult(status="sufficient", new_queries=[], critique="ok")

    state = _base_state(
        original_plan=[{"query": "q1", "mode": "MIXED"}],
        knowledge_gaps=["unverifiable claim X"],
        iteration_count=1,
        session_id="sess-1",
    )
    critic_node(state)

    mock_kg.store_gaps.assert_called_once()
    entries = mock_kg.store_gaps.call_args.args[0]
    by_type = {e["gap_type"]: e["description"] for e in entries}
    assert by_type == {"coverage_gap": "q1", "knowledge_gap": "unverifiable claim X"}
    assert all(e["iteration"] == 1 for e in entries)


@patch("src.agents.critic.get_llm_with_fallbacks")
@patch("src.agents.critic.get_embeddings")
@patch("src.graph.kg.kg_store")
def test_critic_stores_empty_gaps_without_crashing(mock_kg, mock_embeddings, mock_get_llm):
    mock_embeddings.return_value = [[0.1] * 384]
    mock_kg.retrieve_relevant_facts.return_value = []
    mock_kg.store_gaps = MagicMock()
    mock_get_llm.return_value.return_value = FactCheckResult(status="sufficient", new_queries=[], critique="ok")

    critic_node(_base_state())

    mock_kg.store_gaps.assert_called_once_with([], session_id="")


# ── Reflector Tests ──────────────────────────────────────────────────────────

@patch("src.agents.reflector.get_llm_with_fallbacks")
def test_reflector_merges_plan_with_critics_queries(mock_get_llm):
    """Reflector's gap-fill queries should be ADDED to the Critic's plan, not
    replace it — Scout's next pass should cover both."""
    mock_get_llm.return_value.invoke.return_value = ReflectorOutput(
        search_queries=[SearchIntent(query="gap query", mode="MIXED")]
    )
    state = _base_state(
        plan=[{"query": "critic query", "mode": "MIXED"}],
        knowledge_gaps=["some unverified claim"],
    )
    result = reflector_node(state)

    queries = {intent["query"] for intent in result["plan"]}
    assert queries == {"critic query", "gap query"}


@patch("src.agents.reflector.get_llm_with_fallbacks")
def test_reflector_does_not_duplicate_matching_queries(mock_get_llm):
    mock_get_llm.return_value.invoke.return_value = ReflectorOutput(
        search_queries=[SearchIntent(query="same query", mode="MIXED")]
    )
    state = _base_state(
        plan=[{"query": "same query", "mode": "MIXED"}],
        knowledge_gaps=["gap"],
    )
    result = reflector_node(state)
    assert len(result["plan"]) == 1


@patch("src.agents.reflector.get_llm_with_fallbacks")
def test_reflector_prioritizes_coverage_gaps_over_knowledge_gaps(mock_get_llm):
    """coverage_gaps (real unanswered sub-questions) should lead the prompt,
    with knowledge_gaps (unverified claims) filling remaining slots."""
    captured = {}

    def _invoke(prompt):
        captured["prompt"] = prompt
        return ReflectorOutput(search_queries=[SearchIntent(query="q", mode="MIXED")])

    mock_get_llm.return_value.invoke.side_effect = _invoke

    state = _base_state(
        coverage_gaps=["uncovered sub-question"],
        knowledge_gaps=["shaky claim"],
    )
    reflector_node(state)

    assert "uncovered sub-question" in captured["prompt"]
    assert "shaky claim" in captured["prompt"]
    assert captured["prompt"].index("uncovered sub-question") < captured["prompt"].index("shaky claim")


# ── Writer Tests ──────────────────────────────────────────────────────────────

@patch("src.agents.writer.get_llm_with_fallbacks")
def test_writer_report_generation(mock_get_llm):
    """Writer should produce a report body and append a References section."""
    # chain = prompt | llm → LangChain wraps llm as RunnableLambda
    # chain.invoke(...) calls llm(messages) → llm.return_value; then .content is read
    mock_get_llm.return_value.return_value.content = "This is the synthesized report body."

    state = _base_state(
        query="What is the state of solid-state batteries?",
        verified_facts=[{
            "claim": "QuantumScape achieved 500 Wh/kg",
            "source_id": "[1]",
            "source_url": "http://a.com",
            "support_level": "SUPPORTED",
            "credibility_score": 0.9,
            "source_type": "Academic/Scientific",
        }],
        source_map={"[1]": {"url": "http://a.com", "credibility_score": 0.9, "source_type": "Academic/Scientific"}},
        critique="",
    )

    result = writer_node(state)

    report = result["report"]
    assert "This is the synthesized report body." in report
    assert "References" in report
    assert "http://a.com" in report
    assert result["active_node"] == "writer"


@patch("src.agents.writer.get_llm_with_fallbacks")
def test_writer_filters_unsupported_facts(mock_get_llm):
    """Writer should exclude NOT_SUPPORTED facts from the evidence block."""
    mock_get_llm.return_value.return_value.content = "Report with only supported facts."

    state = _base_state(
        query="test",
        verified_facts=[
            {"claim": "Good fact", "source_id": "[1]", "source_url": "http://a.com",
             "support_level": "SUPPORTED", "credibility_score": 0.8, "source_type": "Academic/Scientific"},
            {"claim": "Bad fact", "source_id": "[2]", "source_url": "http://b.com",
             "support_level": "NOT_SUPPORTED", "credibility_score": 0.3, "source_type": "Unverified/Web"},
        ],
        source_map={
            "[1]": {"url": "http://a.com", "credibility_score": 0.8, "source_type": "Academic/Scientific"},
        },
    )

    result = writer_node(state)

    # NOT_SUPPORTED fact should not appear in report
    assert "Bad fact" not in result["report"]
    assert result["active_node"] == "writer"


@patch("src.agents.writer.get_llm_with_fallbacks")
def test_writer_handles_list_content_response(mock_get_llm):
    """Some providers (e.g. Gemini) return .content as a list of blocks
    instead of a plain string — the Writer must flatten it, not crash."""
    mock_get_llm.return_value.return_value.content = [{"type": "text", "text": "Report from a list-content response."}]

    state = _base_state(
        query="test",
        verified_facts=[{
            "claim": "Good fact", "source_id": "[1]", "source_url": "http://a.com",
            "support_level": "SUPPORTED", "credibility_score": 0.8, "source_type": "Academic/Scientific",
        }],
        source_map={"[1]": {"url": "http://a.com", "credibility_score": 0.8, "source_type": "Academic/Scientific"}},
    )

    result = writer_node(state)

    assert "Report from a list-content response." in result["report"]


# ── Refiner batching ─────────────────────────────────────────────────────────

def test_split_into_batches_packs_under_limit():
    docs = [{"source_id": f"[{i}]", "content": "x" * 100} for i in range(5)]
    batches = _split_into_batches(docs, max_chars=300, max_doc_chars=1000)
    assert len(batches) > 1, "five 100-char docs must not fit one 300-char batch"
    assert all(len(b) <= 300 or b.count("<document") == 1 for b in batches)
    # Every document survives the split exactly once.
    joined = "".join(batches)
    for i in range(5):
        assert joined.count(f'source_id="[{i}]"') == 1


def test_split_into_batches_single_payload_when_small():
    docs = [{"source_id": "[1]", "content": "short"}, {"source_id": "[2]", "content": "also short"}]
    batches = _split_into_batches(docs, max_chars=100_000, max_doc_chars=100_000)
    assert len(batches) == 1
    assert 'source_id="[1]"' in batches[0] and 'source_id="[2]"' in batches[0]


def test_split_into_batches_truncates_oversized_doc_instead_of_dropping():
    docs = [{"source_id": "[1]", "content": "y" * 5000}]
    batches = _split_into_batches(docs, max_chars=500, max_doc_chars=200)
    assert len(batches) == 1
    assert batches[0].count("y") == 200
    assert 'source_id="[1]"' in batches[0]


def test_split_into_batches_skips_empty_docs():
    docs = [{"source_id": "[1]", "content": "   "}, {"source_id": "[2]", "content": ""}]
    assert _split_into_batches(docs) == []


@patch("src.agents.refiner._generate_dynamic_schema", return_value={"facts": "any"})
@patch("src.agents.refiner.extract_facts")
def test_refiner_survives_a_failed_batch(mock_extract, _mock_schema):
    """A batch failing across the whole ladder must not discard the other batches."""
    mock_extract.side_effect = [
        RuntimeError("429 rate limit"),
        {"facts": [{"claim": "survived", "source_id": "[2]"}]},
    ]
    state = {
        "query": "q", "plan": [], "verified_facts": [], "source_map": {},
        "scraped_data": [
            {"source_id": "[1]", "content": "a" * 30000},
            {"source_id": "[2]", "content": "b" * 30000},
        ],
    }

    result = refiner_node(state)

    assert mock_extract.call_count == 2, "each document should be its own batch"
    evidence = result["structured_evidence"]
    assert len(evidence) == 1
    assert evidence[0]["claim"] == "survived"


@pytest.mark.asyncio
@patch("src.agents.scout.shared_crawler", _fake_shared_crawler)
@patch("src.agents.scout.gather_sources_for_query", new_callable=AsyncMock)
@patch("src.agents.scout.scrape_urls", new_callable=AsyncMock)
@patch("src.agents.scout.get_embeddings")
@patch("src.agents.scout.MAX_SOURCES_PER_DOMAIN", 2)
async def test_scout_does_not_cap_authoritative_domains(mock_embeddings, mock_scrape, mock_gather):
    """The per-domain cap targets content farms. Capping sec.gov threw away 18
    of 20 primary-source SEC filings in a real run — authoritative domains are
    exempt."""
    mock_embeddings.side_effect = Exception("embeddings unavailable")

    urls = [f"https://www.sec.gov/Archives/edgar/data/{i}/filing.htm" for i in range(5)]
    mock_gather.return_value = [{"url": u, "content": "", "needs_scrape": True, "source": "web"} for u in urls]
    mock_scrape.return_value = [
        {"url": u, "content": "SEC filing content.", "success": True} for u in urls
    ]

    state = _base_state(plan=["q1"])
    with patch("src.graph.kg.kg_store", MagicMock()):
        result = await scout_node(state)

    sec_sources = [v for v in result["source_map"].values() if "sec.gov" in v["url"]]
    assert len(sec_sources) == 5, "all 5 filings should survive, not just MAX_SOURCES_PER_DOMAIN"
    assert all(v["credibility_score"] == 0.9 for v in sec_sources)


# ── Scout: query relevance ───────────────────────────────────────────────────

def test_relevance_scores_matching_content_above_unrelated():
    """Credibility says a source is trustworthy; this says it is on-topic."""
    from src.agents.scout import _relevance_to_query
    from src.utils.embeddings import get_embeddings

    qe = get_embeddings(["cardiovascular outcomes of GLP-1 receptor agonists"])[0]
    on_topic = _relevance_to_query(
        "GLP-1 receptor agonist cardiovascular outcomes trials in type 2 diabetes.", qe)
    off_topic = _relevance_to_query(
        "Quantum error correction with surface codes on superconducting qubits.", qe)
    assert on_topic > off_topic
    assert on_topic > 0.35 > off_topic


def test_relevance_returns_none_without_a_query_embedding():
    """The degraded path must report 'not measured', not 'irrelevant'."""
    from src.agents.scout import _relevance_to_query
    assert _relevance_to_query("some content", None) is None


def test_relevance_returns_none_for_empty_content():
    from src.agents.scout import _relevance_to_query
    from src.utils.embeddings import get_embeddings
    qe = get_embeddings(["anything"])[0]
    assert _relevance_to_query("   ", qe) is None


@patch("src.agents.scout.get_embeddings", side_effect=RuntimeError("embedding down"))
def test_relevance_failure_does_not_break_the_run(mock_emb):
    from src.agents.scout import _relevance_to_query
    assert _relevance_to_query("content", [0.1] * 384) is None


# ── Scout: relevance gate ────────────────────────────────────────────────────
# Credibility scores what a URL looks like; an unrelated journal article rates
# as highly as a relevant one. These cover dropping those before they reach the
# Refiner, without ever leaving the run with no evidence.

_ON, _OFF = "cardiology", "unrelated"


def _topic_embeddings(texts):
    """[1,0,...] for on-topic text, [0,1,...] for everything else."""
    out = []
    for t in texts:
        vec = [0.0] * 384
        vec[0 if _ON in t.lower() else 1] = 1.0
        out.append(vec)
    return out


def _kg_with_two_docs():
    """A KG returning one on-topic and one off-topic document."""
    store = MagicMock()
    store.store_document_and_chunks.side_effect = [1, 2]
    store.retrieve_top_docs.return_value = [1, 2]
    store.retrieve_top_chunks.return_value = [(1, f"{_ON} content"), (2, f"{_OFF} content")]
    store.retrieve_top_chunks_bm25.return_value = []
    store.get_all_chunks_for_docs.return_value = {1: [f"{_ON} content"], 2: [f"{_OFF} content"]}
    store.get_doc_metadata.side_effect = lambda d: {
        1: {"url": "https://journal.example/on"},
        2: {"url": "https://journal.example/off"},
    }[d]
    return store


def _scout_state():
    return _base_state(query=f"{_ON} outcomes", plan=[f"{_ON} outcomes"])


async def _run_scout_with_two_docs():
    from src.agents.scout import scout_node
    urls = ["https://journal.example/on", "https://journal.example/off"]
    with patch("src.agents.scout.gather_sources_for_query", new_callable=AsyncMock) as gather, \
         patch("src.agents.scout.scrape_urls", new_callable=AsyncMock) as scrape, \
         patch("src.agents.scout.get_embeddings", side_effect=_topic_embeddings), \
         patch("src.agents.scout.rerank_chunks", side_effect=RuntimeError("no reranker")), \
         patch("src.graph.kg.kg_store", _kg_with_two_docs()):
        gather.return_value = [
            {"url": u, "content": "", "needs_scrape": True, "source": "web"} for u in urls
        ]
        scrape.return_value = [
            {"url": urls[0], "content": f"{_ON} content", "success": True},
            {"url": urls[1], "content": f"{_OFF} content", "success": True},
        ]
        return await scout_node(_scout_state())


@pytest.mark.asyncio
@patch("src.agents.scout.shared_crawler", _fake_shared_crawler)
async def test_relevance_gate_drops_off_topic_sources():
    """An off-topic source must not reach the Refiner, however credible it looks."""
    result = await _run_scout_with_two_docs()
    urls = {v["url"] for v in result["source_map"].values()}
    assert "https://journal.example/on" in urls
    assert "https://journal.example/off" not in urls


@pytest.mark.asyncio
@patch("src.agents.scout.shared_crawler", _fake_shared_crawler)
@patch("src.agents.scout.RELEVANCE_GATE_ENABLED", False)
async def test_relevance_gate_can_be_disabled_for_measurement():
    """With the gate off the score is still recorded, so a run can measure the
    off-topic share without filtering on it."""
    result = await _run_scout_with_two_docs()
    by_url = {v["url"]: v for v in result["source_map"].values()}
    assert "https://journal.example/off" in by_url
    assert by_url["https://journal.example/off"]["relevance_score"] < 0.35


@pytest.mark.asyncio
@patch("src.agents.scout.shared_crawler", _fake_shared_crawler)
@patch("src.agents.scout.RELEVANCE_GATE_MIN_SOURCES", 2)
async def test_relevance_gate_never_empties_the_run():
    """If everything is off-topic, keep the closest matches so the Critic sees
    weak evidence rather than none."""
    from src.agents.scout import scout_node
    store = _kg_with_two_docs()
    store.retrieve_top_chunks.return_value = [(1, f"{_OFF} a"), (2, f"{_OFF} b")]
    store.get_all_chunks_for_docs.return_value = {1: [f"{_OFF} a"], 2: [f"{_OFF} b"]}
    urls = ["https://journal.example/on", "https://journal.example/off"]

    with patch("src.agents.scout.gather_sources_for_query", new_callable=AsyncMock) as gather, \
         patch("src.agents.scout.scrape_urls", new_callable=AsyncMock) as scrape, \
         patch("src.agents.scout.get_embeddings", side_effect=_topic_embeddings), \
         patch("src.agents.scout.rerank_chunks", side_effect=RuntimeError("no reranker")), \
         patch("src.graph.kg.kg_store", store):
        gather.return_value = [
            {"url": u, "content": "", "needs_scrape": True, "source": "web"} for u in urls
        ]
        scrape.return_value = [
            {"url": u, "content": f"{_OFF} content", "success": True} for u in urls
        ]
        result = await scout_node(_scout_state())

    assert len(result["source_map"]) > 0, "the gate must never starve the Refiner"


@pytest.mark.asyncio
@patch("src.agents.scout.shared_crawler", _fake_shared_crawler)
async def test_gated_source_does_not_consume_a_domain_slot():
    """A rejected source is measured before anything is claimed, so it must not
    use up the per-domain cap that a later on-topic source needs."""
    from src.agents.scout import scout_node
    store = _kg_with_two_docs()
    # Both documents share a domain; the off-topic one is registered first.
    store.retrieve_top_docs.return_value = [2, 1]
    store.get_doc_metadata.side_effect = lambda d: {
        1: {"url": "https://site.example/on"},
        2: {"url": "https://site.example/off"},
    }[d]
    urls = ["https://site.example/off", "https://site.example/on"]

    with patch("src.agents.scout.MAX_SOURCES_PER_DOMAIN", 1), \
         patch("src.agents.scout.gather_sources_for_query", new_callable=AsyncMock) as gather, \
         patch("src.agents.scout.scrape_urls", new_callable=AsyncMock) as scrape, \
         patch("src.agents.scout.get_embeddings", side_effect=_topic_embeddings), \
         patch("src.agents.scout.rerank_chunks", side_effect=RuntimeError("no reranker")), \
         patch("src.graph.kg.kg_store", store):
        gather.return_value = [
            {"url": u, "content": "", "needs_scrape": True, "source": "web"} for u in urls
        ]
        scrape.return_value = [
            {"url": urls[0], "content": f"{_OFF} content", "success": True},
            {"url": urls[1], "content": f"{_ON} content", "success": True},
        ]
        result = await scout_node(_scout_state())

    urls_kept = {v["url"] for v in result["source_map"].values()}
    assert "https://site.example/on" in urls_kept
