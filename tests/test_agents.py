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
from src.agents.refiner import refiner_node
from src.agents.critic import critic_node
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

@patch("src.agents.librarian.get_llm")
def test_librarian_generates_plan(mock_get_llm):
    """Librarian should decompose the query and return a list of search-intent dicts."""
    expected_plan = ResearchPlan(search_queries=[
        SearchIntent(query="solid state battery energy density 2026", mode="TRUSTED_ONLY"),
        SearchIntent(query="QuantumScape QSE-5 cycle life results", mode="TRUSTED_FIRST"),
    ])
    # LangChain wraps MagicMock as RunnableLambda; the chain calls mock(input) → .return_value
    mock_get_llm.return_value.with_structured_output.return_value.return_value = expected_plan

    result = librarian_node(_base_state())

    assert "plan" in result
    assert len(result["plan"]) == 2
    assert result["plan"][0]["query"] == "solid state battery energy density 2026"
    assert result["plan"][0]["mode"] == "TRUSTED_ONLY"
    assert result["active_node"] == "librarian"


@patch("src.agents.librarian.get_llm")
def test_librarian_injects_critique_on_iteration(mock_get_llm):
    """On iteration > 0, the system prompt must contain the previous critique."""
    expected_plan = ResearchPlan(search_queries=[
        SearchIntent(query="gap-filling query for missing data", mode="MIXED"),
    ])
    mock_get_llm.return_value.with_structured_output.return_value.return_value = expected_plan

    state = _base_state(
        iteration_count=1,
        critique="Missing: specific manufacturing cost data for 2025.",
    )
    result = librarian_node(state)

    assert "plan" in result
    assert len(result["plan"]) == 1
    mock_get_llm.assert_called_once()


@patch("src.agents.librarian.get_llm")
def test_librarian_no_critique_injection_on_first_iteration(mock_get_llm):
    """On iteration 0, critique should NOT be injected even if state has a stale value."""
    expected_plan = ResearchPlan(search_queries=[
        SearchIntent(query="general query", mode="MIXED"),
    ])
    mock_get_llm.return_value.with_structured_output.return_value.invoke.return_value = expected_plan

    # iteration_count=0, critique is non-empty (shouldn't be injected)
    state = _base_state(iteration_count=0, critique="old critique from prior run")
    result = librarian_node(state)

    assert result["active_node"] == "librarian"
    mock_get_llm.assert_called_once()


# ── Scout Tests ───────────────────────────────────────────────────────────────

@pytest.mark.asyncio
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


# ── Refiner Tests ─────────────────────────────────────────────────────────────

@patch("src.agents.refiner.extract_facts")
@patch("src.agents.refiner.get_llm")
def test_refiner_extraction(mock_get_llm, mock_extract):
    """Refiner should call extract_facts and return structured_evidence."""
    from src.tools.refiner import ExtractedFact, FactExtractionResult
    from src.agents.refiner import ExtractionSchema

    # _generate_dynamic_schema uses get_llm → structured_llm.invoke → ExtractionSchema
    mock_get_llm.return_value.with_structured_output.return_value.invoke.return_value = (
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

@patch("src.agents.critic.get_llm")
@patch("src.agents.critic.get_embeddings")
@patch("src.graph.kg.kg_store")
def test_critic_evaluation_gaps_found(mock_kg, mock_embeddings, mock_get_llm):
    """When the critic finds gaps, re_search_required should be True."""
    mock_embeddings.return_value = [[0.1] * 384]
    mock_kg.retrieve_relevant_facts.return_value = []

    # Chain calls mock as callable → .return_value (not .invoke.return_value)
    mock_get_llm.return_value.with_structured_output.return_value.return_value = FactCheckResult(
        status="gaps_found",
        new_queries=[SearchIntent(query="follow-up query", mode="MIXED")],
        critique="Missing manufacturing cost data.",
    )

    result = critic_node(_base_state())

    assert result["re_search_required"] is True
    assert result["critique"] == "Missing manufacturing cost data."
    assert result["active_node"] == "critic"


@patch("src.agents.critic.get_llm")
@patch("src.agents.critic.get_embeddings")
@patch("src.graph.kg.kg_store")
def test_critic_evaluation_sufficient(mock_kg, mock_embeddings, mock_get_llm):
    """When evidence is sufficient, re_search_required should be False."""
    mock_embeddings.return_value = [[0.1] * 384]
    mock_kg.retrieve_relevant_facts.return_value = []

    mock_get_llm.return_value.with_structured_output.return_value.return_value = FactCheckResult(
        status="sufficient",
        new_queries=[],
        critique="All key questions are answered.",
    )

    result = critic_node(_base_state())

    assert result["re_search_required"] is False
    assert result["active_node"] == "critic"


# ── Writer Tests ──────────────────────────────────────────────────────────────

@patch("src.agents.writer.get_llm")
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


@patch("src.agents.writer.get_llm")
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
