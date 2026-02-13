"""
Project Argus - Tool Tests

Tests for the search, scraper, and scout pipeline.
Run with: python -m pytest tests/test_tools.py -v
"""

import pytest
import asyncio
from src.tools.search import search_ddg, _is_junk_domain
from src.tools.scraper import scrape_urls
from src.tools.scout import run_scout


# ── Search Tests ─────────────────────────────────────────────────────────────


class TestSearchDDG:
    """Tests for the DuckDuckGo search wrapper."""

    def test_junk_domain_filter(self):
        """Known junk domains should be detected."""
        assert _is_junk_domain("https://www.pinterest.com/pin/12345") is True
        assert _is_junk_domain("https://quora.com/some-question") is True
        assert _is_junk_domain("https://www.reddit.com/r/test") is True
        assert _is_junk_domain("https://medium.com/article") is True

    def test_clean_domain_passes(self):
        """Legitimate research domains should pass the filter."""
        assert _is_junk_domain("https://arxiv.org/abs/2024.01234") is False
        assert _is_junk_domain("https://nature.com/articles/s41586") is False
        assert _is_junk_domain("https://en.wikipedia.org/wiki/Battery") is False

    @pytest.mark.asyncio
    async def test_search_returns_urls(self):
        """search_ddg should return a non-empty list of URL strings."""
        urls = await search_ddg("solid state batteries 2026", max_results=3)

        assert isinstance(urls, list)
        assert len(urls) > 0
        assert all(isinstance(u, str) for u in urls)
        assert all(u.startswith("http") for u in urls)

    @pytest.mark.asyncio
    async def test_search_filters_junk(self):
        """Returned URLs should not contain any junk domains."""
        from src.config import JUNK_DOMAINS

        urls = await search_ddg("python programming tutorial", max_results=5)
        for url in urls:
            assert not _is_junk_domain(url), f"Junk domain leaked through: {url}"

    @pytest.mark.asyncio
    async def test_search_respects_max_results(self):
        """Should return at most max_results URLs."""
        urls = await search_ddg("artificial intelligence", max_results=3)
        assert len(urls) <= 3

    @pytest.mark.asyncio
    async def test_search_handles_empty_query(self):
        """An empty or garbage query should not crash."""
        urls = await search_ddg("")
        assert isinstance(urls, list)


# ── Scraper Tests ────────────────────────────────────────────────────────────


class TestScraper:
    """Tests for the Crawl4AI scraper."""

    @pytest.mark.asyncio
    async def test_scrape_returns_structured_results(self):
        """Each result should have url, content, success, and error keys."""
        results = await scrape_urls(["https://httpbin.org/html"])

        assert len(results) == 1
        result = results[0]
        assert "url" in result
        assert "content" in result
        assert "success" in result
        assert "error" in result

    @pytest.mark.asyncio
    async def test_scrape_successful_page(self):
        """A known good URL should return content."""
        results = await scrape_urls(["https://httpbin.org/html"])
        result = results[0]

        assert result["success"] is True
        assert len(result["content"]) > 0
        assert result["error"] is None

    @pytest.mark.asyncio
    async def test_scrape_handles_bad_url(self):
        """A bad URL should return a structured error, not crash."""
        results = await scrape_urls(["https://thisdomaindoesnotexist12345.com"])
        result = results[0]

        assert result["success"] is False
        assert result["error"] is not None

    @pytest.mark.asyncio
    async def test_scrape_empty_list(self):
        """An empty URL list should return an empty result list."""
        results = await scrape_urls([])
        assert results == []

    @pytest.mark.asyncio
    async def test_scrape_concurrent_multiple(self):
        """Multiple URLs should be scraped concurrently."""
        urls = [
            "https://httpbin.org/html",
            "https://example.com",
        ]
        results = await scrape_urls(urls)

        assert len(results) == 2
        assert all("url" in r for r in results)


# ── Scout Pipeline Tests (End-to-End) ───────────────────────────────────────


class TestRunScout:
    """End-to-end tests for the full scout pipeline."""

    @pytest.mark.asyncio
    async def test_run_scout_returns_results(self):
        """
        The Phase 1 deliverable test:
        run_scout should return a list of scraped results.
        """
        results = await run_scout("solid state batteries 2026", max_results=3)

        assert isinstance(results, list)
        assert len(results) > 0

        # Each result should be a structured dict
        for r in results:
            assert "url" in r
            assert "content" in r
            assert "success" in r
            assert "error" in r

        # At least one result should be successful
        successful = [r for r in results if r["success"]]
        assert len(successful) > 0, "No pages were successfully scraped"

        # Successful results should have actual content
        for r in successful:
            assert len(r["content"]) > 100, (
                f"Content too short for {r['url']}: {len(r['content'])} chars"
            )

    @pytest.mark.asyncio
    async def test_run_scout_respects_max_results(self):
        """Should not return more results than max_results."""
        results = await run_scout("quantum computing", max_results=2)
        assert len(results) <= 2


# ── Refiner Tests (LangExtract) ─────────────────────────────────────────────


from src.tools.refiner import extract_facts


# Sample text used across refiner tests
_REFINER_SAMPLE = (
    "QuantumScape announced on January 15, 2026 that their prototype cell "
    "achieved an energy density of 500 Wh/kg. The estimated cost per kWh "
    "has dropped to $80. A solid-state battery replaces the liquid "
    "electrolyte with a solid material."
)


class TestExtractFacts:
    """Tests for the LangExtract refiner tool."""

    def test_import_works(self):
        """The refiner module should import without errors."""
        assert callable(extract_facts)

    def test_extract_returns_facts_key(self):
        """extract_facts should return a dict with a 'facts' list."""
        schema = {"metrics": "numeric value"}
        result = extract_facts(_REFINER_SAMPLE, schema)

        assert isinstance(result, dict)
        assert "facts" in result
        assert isinstance(result["facts"], list)

    def test_fact_structure(self):
        """Each fact should have class, text, source_span, and attributes."""
        schema = {"metrics": "numeric value", "dates": "date"}
        result = extract_facts(_REFINER_SAMPLE, schema)

        for fact in result["facts"]:
            assert "class" in fact, "Missing 'class' key"
            assert "text" in fact, "Missing 'text' key"
            assert "source_span" in fact, "Missing 'source_span' key"
            assert "start" in fact["source_span"]
            assert "end" in fact["source_span"]
            assert "attributes" in fact, "Missing 'attributes' key"

    def test_source_span_points_to_text(self):
        """source_span should reference real positions in the input text."""
        schema = {"metrics": "numeric value"}
        result = extract_facts(_REFINER_SAMPLE, schema)

        for fact in result["facts"]:
            start = fact["source_span"]["start"]
            end = fact["source_span"]["end"]
            # Spans should be valid character indices
            if start is not None and end is not None:
                assert start >= 0, f"Negative start index: {start}"
                assert end >= start, f"end ({end}) < start ({start})"
                assert end <= len(_REFINER_SAMPLE), (
                    f"end ({end}) exceeds text length ({len(_REFINER_SAMPLE)})"
                )

    def test_empty_text_does_not_crash(self):
        """Passing empty text should return a result, not raise."""
        schema = {"dates": "date"}
        result = extract_facts("", schema)

        assert isinstance(result, dict)
        assert "facts" in result

    def test_raw_jsonl_present(self):
        """Result should include a raw_jsonl string for archiving."""
        schema = {"metrics": "numeric value"}
        result = extract_facts(_REFINER_SAMPLE, schema)

        assert "raw_jsonl" in result
        assert isinstance(result["raw_jsonl"], str)

