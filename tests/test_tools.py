"""
Project Argus - Tool Tests

Tests for the search, scraper, and scout pipeline.
Run with: python -m pytest tests/test_tools.py -v
"""

import pytest
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock
from src.tools.search import search_ddg, _is_junk_domain
from src.tools.scraper import scrape_urls, shared_crawler
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


class TestSharedCrawler:
    """shared_crawler()'s browser close is time-boxed: Playwright's close
    handshake can hang indefinitely with no error, which has stalled a whole
    research job with no log output. These mock AsyncWebCrawler directly so
    the hang is simulated, not a real (slow, flaky) browser interaction."""

    @pytest.mark.asyncio
    @patch("src.tools.scraper.AsyncWebCrawler")
    async def test_starts_and_closes_normally(self, mock_cls):
        mock_crawler = MagicMock()
        mock_crawler.start = AsyncMock()
        mock_crawler.close = AsyncMock()
        mock_cls.return_value = mock_crawler

        async with shared_crawler() as crawler:
            assert crawler is mock_crawler

        mock_crawler.start.assert_awaited_once()
        mock_crawler.close.assert_awaited_once()

    @pytest.mark.asyncio
    @patch("src.tools.scraper.CRAWLER_CLOSE_TIMEOUT", 0.05)
    @patch("src.tools.scraper.AsyncWebCrawler")
    async def test_survives_a_hung_close(self, mock_cls):
        """A close() that never returns must not block the caller forever."""
        async def _hang_forever():
            await asyncio.sleep(10)

        mock_crawler = MagicMock()
        mock_crawler.start = AsyncMock()
        mock_crawler.close = AsyncMock(side_effect=_hang_forever)
        mock_cls.return_value = mock_crawler

        async def _use_it():
            async with shared_crawler() as crawler:
                return crawler

        result = await asyncio.wait_for(_use_it(), timeout=2.0)
        assert result is mock_crawler

    @pytest.mark.asyncio
    @patch("src.tools.scraper.AsyncWebCrawler")
    async def test_swallows_close_errors(self, mock_cls):
        """A close() that raises shouldn't crash a run whose scraping already succeeded."""
        mock_crawler = MagicMock()
        mock_crawler.start = AsyncMock()
        mock_crawler.close = AsyncMock(side_effect=RuntimeError("boom"))
        mock_cls.return_value = mock_crawler

        async with shared_crawler() as crawler:
            assert crawler is mock_crawler
        # No exception propagated out of the `async with` — reaching here is the assertion.


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


from src.tools.refiner import extract_facts, ExtractedFact, FactExtractionResult


# Sample text used across refiner tests
_REFINER_SAMPLE = (
    "QuantumScape announced on January 15, 2026 that their prototype cell "
    "achieved an energy density of 500 Wh/kg. The estimated cost per kWh "
    "has dropped to $80. A solid-state battery replaces the liquid "
    "electrolyte with a solid material."
)


def _build_mock_extraction_result() -> FactExtractionResult:
    """Returns a real FactExtractionResult for use as a mock LLM response."""
    return FactExtractionResult(facts=[
        ExtractedFact(
            extraction_class="metrics",
            claim="500 Wh/kg energy density",
            source_excerpt="achieved an energy density of 500 Wh/kg",
            source_id="[1]",
            attributes={"unit": "Wh/kg", "value": "500"},
        )
    ])


class TestExtractFacts:
    """Tests for the refiner tool."""

    def test_import_works(self):
        """The refiner module should import without errors."""
        assert callable(extract_facts)

    @patch("src.tools.refiner.get_llm_with_fallbacks")
    def test_extract_returns_facts_key(self, mock_get_llm):
        """extract_facts should return a dict with a 'facts' list."""
        mock_get_llm.return_value.invoke.return_value = (
            _build_mock_extraction_result()
        )
        result = extract_facts(_REFINER_SAMPLE, {"metrics": "numeric value"})

        assert isinstance(result, dict)
        assert "facts" in result
        assert isinstance(result["facts"], list)

    @patch("src.tools.refiner.get_llm_with_fallbacks")
    def test_fact_structure(self, mock_get_llm):
        """Each fact should have class, claim, source_excerpt, source_span, and attributes."""
        mock_get_llm.return_value.invoke.return_value = (
            _build_mock_extraction_result()
        )
        result = extract_facts(_REFINER_SAMPLE, {"metrics": "numeric value"})

        assert len(result["facts"]) > 0
        for fact in result["facts"]:
            assert "class" in fact
            assert "claim" in fact
            assert "source_excerpt" in fact
            assert "source_span" in fact
            assert "start" in fact["source_span"]
            assert "end" in fact["source_span"]
            assert "attributes" in fact

    @patch("src.tools.refiner.get_llm_with_fallbacks")
    def test_source_span_points_to_text(self, mock_get_llm):
        """source_span indices, when not None, should be valid offsets."""
        mock_get_llm.return_value.invoke.return_value = (
            _build_mock_extraction_result()
        )
        result = extract_facts(_REFINER_SAMPLE, {"metrics": "numeric value"})

        for fact in result["facts"]:
            start = fact["source_span"]["start"]
            end = fact["source_span"]["end"]
            if start is not None and end is not None:
                assert start >= 0
                assert end >= start

    @patch("src.tools.refiner.get_llm_with_fallbacks")
    def test_empty_text_does_not_crash(self, mock_get_llm):
        """Passing empty text should return a result, not raise."""
        mock_get_llm.return_value.invoke.return_value = (
            FactExtractionResult(facts=[])
        )
        result = extract_facts("", {"dates": "date"})

        assert isinstance(result, dict)
        assert "facts" in result
        assert result["facts"] == []

    @patch("src.tools.refiner.get_llm_with_fallbacks")
    def test_raw_jsonl_present(self, mock_get_llm):
        """Result should include a raw_jsonl key."""
        mock_get_llm.return_value.invoke.return_value = (
            _build_mock_extraction_result()
        )
        result = extract_facts(_REFINER_SAMPLE, {"metrics": "numeric value"})

        assert "raw_jsonl" in result
        assert isinstance(result["raw_jsonl"], str)


