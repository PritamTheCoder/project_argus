"""
Project Argus - Search provider parser tests (pure, no network)

Exercises the PURE parse functions that turn raw API payloads into
normalized SearchResults, plus the candidate shaping logic.
"""

import pytest
from unittest.mock import patch, AsyncMock, MagicMock

from src.tools.providers import (
    SearchResult,
    _parse_semantic_scholar,
    semantic_scholar_search,
    _parse_arxiv,
    _parse_crossref,
    _parse_brave,
    _ddgs_to_results,
    _clean_text,
    _parse_exa,
    _parse_europe_pmc,
    europe_pmc_search,
    _parse_edgar,
    _edgar_filing_url,
    _edgar_primary_doc_url,
    _parse_form_d_xml,
    _form_d_text,
    _enrich_edgar_filings,
    _is_pooled_vehicle,
    resolve_cik,
    EXA_CATEGORIES,
    web_search,
    exa_search,
    edgar_search,
)


def test_clean_text_strips_tags_and_whitespace():
    assert _clean_text("<jats:p>Hello   world</jats:p>") == "Hello world"
    assert _clean_text(None) == ""


def test_parse_semantic_scholar_prefers_open_access_pdf():
    data = {"data": [{
        "title": "SSB review",
        "abstract": "A detailed abstract about solid-state batteries.",
        "year": 2024,
        "url": "http://semanticscholar.org/paper/1",
        "authors": [{"name": "Alice"}, {"name": "Bob"}],
        "citationCount": 12,
        "openAccessPdf": {"url": "http://example.org/paper.pdf"},
    }]}
    results = _parse_semantic_scholar(data)
    assert len(results) == 1
    r = results[0]
    assert r.url == "http://example.org/paper.pdf"  # PDF overrides landing page
    assert r.source == "semantic_scholar"
    assert r.as_of_date == "2024"
    assert r.authors == ["Alice", "Bob"]
    assert r.credibility_hint == 0.9
    assert r.source_type_hint == "Academic/Scientific"


def test_parse_arxiv_atom():
    xml = """<feed xmlns="http://www.w3.org/2005/Atom">
      <entry>
        <title>Quantum Error Correction</title>
        <summary>We present a new QEC scheme.</summary>
        <id>http://arxiv.org/abs/2401.00001</id>
        <published>2024-01-02T00:00:00Z</published>
        <author><name>Carol</name></author>
      </entry>
    </feed>"""
    results = _parse_arxiv(xml)
    assert len(results) == 1
    r = results[0]
    assert r.title == "Quantum Error Correction"
    assert r.url == "http://arxiv.org/abs/2401.00001"
    assert r.snippet == "We present a new QEC scheme."
    assert r.as_of_date == "2024-01-02"
    assert r.source == "arxiv"
    assert r.authors == ["Carol"]


def test_parse_arxiv_bad_xml_returns_empty():
    assert _parse_arxiv("<not valid") == []


def test_parse_crossref_strips_jats_and_builds_doi_url():
    data = {"message": {"items": [{
        "title": ["Hybrid Retrieval for RAG"],
        "abstract": "<jats:p>BM25 plus dense retrieval.</jats:p>",
        "DOI": "10.1000/xyz",
        "issued": {"date-parts": [[2023, 5, 1]]},
        "author": [{"given": "Jane", "family": "Doe"}],
        "container-title": ["Journal of IR"],
    }]}}
    results = _parse_crossref(data)
    assert len(results) == 1
    r = results[0]
    assert r.title == "Hybrid Retrieval for RAG"
    assert r.url == "https://doi.org/10.1000/xyz"
    assert r.snippet == "BM25 plus dense retrieval."
    assert r.as_of_date == "2023"
    assert r.authors == ["Jane Doe"]
    assert r.extra["venue"] == "Journal of IR"


def test_parse_brave():
    data = {"web": {"results": [
        {"title": "Market report", "url": "http://news.example/report", "description": "desc", "page_age": "2024-03-01T00:00:00"},
        {"title": "no url", "url": "", "description": "skip"},
    ]}}
    results = _parse_brave(data)
    assert len(results) == 1
    assert results[0].url == "http://news.example/report"
    assert results[0].as_of_date == "2024-03-01"
    assert results[0].source == "web"


def test_ddgs_conversion():
    raw = [{"title": "T", "href": "http://d.example", "body": "snippet"}, {"title": "x", "href": ""}]
    results = _ddgs_to_results(raw)
    assert len(results) == 1
    assert results[0].url == "http://d.example"
    assert results[0].source == "web"


# ── Candidate shaping ────────────────────────────────────────────────────────

def test_academic_candidate_no_scrape_needed():
    r = SearchResult(title="t", url="http://p", snippet="abstract body", source="arxiv")
    c = r.to_candidate()
    assert c["needs_scrape"] is False
    assert c["content"] == "abstract body"


def test_web_candidate_needs_scrape():
    r = SearchResult(title="t", url="http://w", snippet="short snippet", source="web")
    c = r.to_candidate()
    assert c["needs_scrape"] is True
    assert c["content"] == ""  # web content comes from the scraper, not the snippet


def test_academic_candidate_without_abstract_needs_scrape():
    r = SearchResult(title="t", url="http://p", snippet="", source="semantic_scholar")
    c = r.to_candidate()
    assert c["needs_scrape"] is True  # no abstract → must fetch


# ── web_search: credibility sort before truncation ──────────────────────────

@pytest.mark.asyncio
@patch("src.tools.providers.EXA_API_KEY", "")
@patch("src.tools.providers.BRAVE_API_KEY", "")
@patch("src.tools.providers._ddg_web_search", new_callable=AsyncMock)
async def test_web_search_sorts_by_credibility_before_truncating(mock_ddg):
    """DDG over-fetches; a higher-credibility result ranked lower by DDG must
    still survive truncation to `limit` over a generic-tier one ranked first."""
    mock_ddg.return_value = [
        SearchResult(url="http://randomblog.info/x", title="junk", source="web"),
        SearchResult(url="http://reuters.com/y", title="news", source="web"),
    ]
    results = await web_search("some query", limit=1)
    assert [r.url for r in results] == ["http://reuters.com/y"]


@pytest.mark.asyncio
@patch("src.tools.providers.EXA_API_KEY", "")
@patch("src.tools.providers.BRAVE_API_KEY", "")
@patch("src.tools.providers._ddg_web_search", new_callable=AsyncMock)
async def test_web_search_stable_sort_preserves_rank_within_tier(mock_ddg):
    """Same-tier results keep DDG's original relevance order (stable sort)."""
    mock_ddg.return_value = [
        SearchResult(url="http://siteb.com/1", title="b", source="web"),
        SearchResult(url="http://sitea.com/1", title="a", source="web"),
    ]
    results = await web_search("q", limit=2)
    assert [r.url for r in results] == ["http://siteb.com/1", "http://sitea.com/1"]


@pytest.mark.asyncio
@patch("src.tools.providers.EXA_API_KEY", "")
@patch("src.tools.providers.BRAVE_API_KEY", "")
@patch("src.tools.providers._ddg_web_search", new_callable=AsyncMock)
async def test_web_search_still_filters_junk_domains(mock_ddg):
    mock_ddg.return_value = [
        SearchResult(url="http://reddit.com/thread", title="junk", source="web"),
        SearchResult(url="http://reuters.com/y", title="news", source="web"),
    ]
    results = await web_search("q", limit=5)
    assert [r.url for r in results] == ["http://reuters.com/y"]


# ── Exa ──────────────────────────────────────────────────────────────────────

def test_parse_exa_extracts_text_and_date():
    data = {"results": [{
        "title": "SpaceX raises Series N",
        "url": "https://example.com/a",
        "publishedDate": "2026-03-02T00:00:00.000Z",
        "author": "Jane Doe",
        "text": "SpaceX announced a funding round.",
        "score": 0.42,
    }]}
    results = _parse_exa(data)
    assert len(results) == 1
    r = results[0]
    assert r.url == "https://example.com/a"
    assert r.text == "SpaceX announced a funding round."
    assert r.as_of_date == "2026-03-02"
    assert r.authors == ["Jane Doe"]
    assert r.source == "exa"


def test_parse_exa_skips_results_without_url():
    assert _parse_exa({"results": [{"title": "no url"}]}) == []


def test_parse_exa_handles_empty_and_garbage():
    assert _parse_exa({}) == []
    assert _parse_exa({"results": None}) == []


def test_exa_result_with_text_needs_no_scrape():
    """Text Exa already fetched means the scraper can be skipped entirely."""
    r = SearchResult(url="http://x", text="full page body", source="exa")
    c = r.to_candidate()
    assert c["needs_scrape"] is False
    assert c["content"] == "full page body"


def test_exa_result_without_text_still_needs_scrape():
    r = SearchResult(url="http://x", snippet="just a snippet", source="exa")
    assert r.to_candidate()["needs_scrape"] is True


@pytest.mark.asyncio
@patch("src.tools.providers.EXA_API_KEY", "")
async def test_exa_search_without_key_returns_empty():
    """No key must degrade to empty, so web_search falls through to the next rung."""
    assert await exa_search("anything") == []


# ── SEC EDGAR ────────────────────────────────────────────────────────────────

def test_edgar_filing_url_strips_zeros_and_dashes():
    url = _edgar_filing_url("0001464202-09-000001", "0001464202")
    assert url == (
        "https://www.sec.gov/Archives/edgar/data/1464202/"
        "000146420209000001/0001464202-09-000001-index.htm"
    )


def test_edgar_filing_url_empty_on_missing_parts():
    assert _edgar_filing_url("", "0001464202") == ""
    assert _edgar_filing_url("0001464202-09-000001", "") == ""


def test_parse_edgar_builds_candidates_with_gov_credibility():
    data = {"hits": {"hits": [{"_source": {
        "adsh": "0001464202-09-000001",
        "ciks": ["0001464202"],
        "display_names": ["Founders Fund SpaceX Fund, LP  (CIK 0001464202)"],
        "form": "D",
        "file_date": "2009-05-15",
    }}]}}
    results = _parse_edgar(data)
    assert len(results) == 1
    r = results[0]
    assert r.source == "sec_edgar"
    assert r.credibility_hint == 0.9
    assert r.source_type_hint == "Government/Institutional"
    assert r.as_of_date == "2009-05-15"
    assert "sec.gov" in r.url


def test_parse_edgar_skips_hits_without_accession_or_cik():
    data = {"hits": {"hits": [{"_source": {"form": "D", "ciks": []}}]}}
    assert _parse_edgar(data) == []


def test_parse_edgar_handles_empty_and_garbage():
    assert _parse_edgar({}) == []
    assert _parse_edgar({"hits": None}) == []


@pytest.mark.asyncio
async def test_edgar_search_empty_query_returns_empty():
    assert await edgar_search("   ") == []


# ── web_search provider ladder ───────────────────────────────────────────────

@pytest.mark.asyncio
@patch("src.tools.providers.EXA_API_KEY", "exa-key")
@patch("src.tools.providers.exa_search", new_callable=AsyncMock)
async def test_web_search_prefers_exa_when_configured(mock_exa):
    mock_exa.return_value = [SearchResult(url="http://reuters.com/x", source="exa")]
    results = await web_search("q", limit=5)
    assert [r.url for r in results] == ["http://reuters.com/x"]
    mock_exa.assert_awaited_once()


@pytest.mark.asyncio
@patch("src.tools.providers.EXA_API_KEY", "exa-key")
@patch("src.tools.providers.BRAVE_API_KEY", "brave-key")
@patch("src.tools.providers._ddg_web_search", new_callable=AsyncMock)
@patch("src.tools.providers._brave_web_search", new_callable=AsyncMock)
@patch("src.tools.providers.exa_search", new_callable=AsyncMock)
async def test_web_search_falls_back_exa_to_brave(mock_exa, mock_brave, mock_ddg):
    """An Exa outage (empty result) must fall through to Brave, not lose the query."""
    mock_exa.return_value = []
    mock_brave.return_value = [SearchResult(url="http://reuters.com/b", source="web")]
    results = await web_search("q", limit=5)
    assert [r.url for r in results] == ["http://reuters.com/b"]
    mock_ddg.assert_not_awaited()


@pytest.mark.asyncio
@patch("src.tools.providers.EXA_API_KEY", "exa-key")
@patch("src.tools.providers.BRAVE_API_KEY", "brave-key")
@patch("src.tools.providers._ddg_web_search", new_callable=AsyncMock)
@patch("src.tools.providers._brave_web_search", new_callable=AsyncMock)
@patch("src.tools.providers.exa_search", new_callable=AsyncMock)
async def test_web_search_falls_all_the_way_to_ddg(mock_exa, mock_brave, mock_ddg):
    """Both paid rungs down → DuckDuckGo still answers."""
    mock_exa.return_value = []
    mock_brave.return_value = []
    mock_ddg.return_value = [SearchResult(url="http://example.com/d", source="web")]
    results = await web_search("q", limit=5)
    assert [r.url for r in results] == ["http://example.com/d"]


@pytest.mark.asyncio
@patch("src.tools.providers.EXA_API_KEY", "")
@patch("src.tools.providers.BRAVE_API_KEY", "")
@patch("src.tools.providers._ddg_web_search", new_callable=AsyncMock)
async def test_web_search_with_no_keys_is_unchanged(mock_ddg):
    """With nothing configured, behaviour is exactly the pre-Exa DuckDuckGo path."""
    mock_ddg.return_value = [SearchResult(url="http://example.com/d", source="web")]
    results = await web_search("q", limit=5)
    assert [r.url for r in results] == ["http://example.com/d"]
    mock_ddg.assert_awaited_once()


def test_exa_categories_match_canonical_reference():
    """Exa rejects unknown categories; the guide Exa generates for users lists
    some that its own API docs don't accept (e.g. "research paper", "pdf")."""
    assert EXA_CATEGORIES == {
        "company", "people", "publication", "news", "personal site", "financial report",
    }


@pytest.mark.asyncio
@patch("src.tools.providers.EXA_API_KEY", "k")
@patch("src.tools.providers.EXA_CACHE_TTL_HOURS", 0)
@patch("src.tools.providers._request", new_callable=AsyncMock)
async def test_exa_search_sends_type_and_nested_contents(mock_req):
    """`text`/`highlights` must be nested under `contents`, and a search type sent."""
    mock_req.return_value = MagicMock(json=lambda: {"results": []})
    await exa_search("q", limit=3)
    payload = mock_req.await_args.kwargs["json"]
    assert payload["type"]                      # a search type is always sent
    assert payload["numResults"] == 3
    assert "contents" in payload
    assert "text" not in payload                # never top-level on /search
    assert "highlights" not in payload


@pytest.mark.asyncio
@patch("src.tools.providers.EXA_API_KEY", "k")
@patch("src.tools.providers.EXA_CACHE_TTL_HOURS", 0)
@patch("src.tools.providers._request", new_callable=AsyncMock)
async def test_exa_search_drops_invalid_category(mock_req):
    """An invalid category degrades to an ordinary search rather than a 400."""
    mock_req.return_value = MagicMock(json=lambda: {"results": []})
    await exa_search("q", category="research paper")   # not a real Exa category
    assert "category" not in mock_req.await_args.kwargs["json"]


@pytest.mark.asyncio
@patch("src.tools.providers.EXA_API_KEY", "k")
@patch("src.tools.providers.EXA_CACHE_TTL_HOURS", 0)
@patch("src.tools.providers._request", new_callable=AsyncMock)
async def test_exa_search_passes_valid_category(mock_req):
    mock_req.return_value = MagicMock(json=lambda: {"results": []})
    await exa_search("q", category="financial report")
    assert mock_req.await_args.kwargs["json"]["category"] == "financial report"


def test_parse_exa_falls_back_to_highlights_when_no_text():
    """highlights mode returns excerpts as a list — they land in `text` too."""
    data = {"results": [{
        "url": "https://example.com/a",
        "highlights": ["first excerpt", "second excerpt"],
    }]}
    r = _parse_exa(data)[0]
    assert "first excerpt" in r.text and "second excerpt" in r.text
    assert r.to_candidate()["needs_scrape"] is False


# ── Exa cost controls ────────────────────────────────────────────────────────

@pytest.mark.asyncio
@patch("src.tools.providers.EXA_API_KEY", "k")
@patch("src.tools.providers.EXA_MAX_RESULTS", 5)
@patch("src.tools.providers.EXA_CACHE_TTL_HOURS", 0)
@patch("src.tools.providers._request", new_callable=AsyncMock)
async def test_exa_caps_results_at_configured_max(mock_req):
    """Contents bill per page, so we never request more results than we use."""
    mock_req.return_value = MagicMock(json=lambda: {"results": []})
    await exa_search("q", limit=20)
    assert mock_req.await_args.kwargs["json"]["numResults"] == 5


@pytest.mark.asyncio
@patch("src.tools.providers.EXA_API_KEY", "k")
@patch("src.tools.providers.EXA_CACHE_TTL_HOURS", 24)
@patch("src.utils.cache.set_cached_search")
@patch("src.utils.cache.get_cached_search")
@patch("src.tools.providers._request", new_callable=AsyncMock)
async def test_exa_cache_hit_skips_the_billed_request(mock_req, mock_get, mock_set):
    """A cache hit must not touch the network — that's the whole point."""
    mock_get.return_value = '{"results": [{"url": "https://cached.example/a"}]}'
    results = await exa_search("q")
    assert [r.url for r in results] == ["https://cached.example/a"]
    mock_req.assert_not_awaited()
    mock_set.assert_not_called()


@pytest.mark.asyncio
@patch("src.tools.providers.EXA_API_KEY", "k")
@patch("src.tools.providers.EXA_CACHE_TTL_HOURS", 24)
@patch("src.utils.cache.set_cached_search")
@patch("src.utils.cache.get_cached_search", return_value=None)
@patch("src.tools.providers._request", new_callable=AsyncMock)
async def test_exa_caches_successful_response(mock_req, mock_get, mock_set):
    mock_req.return_value = MagicMock(json=lambda: {"results": [{"url": "https://x.example/a"}]})
    await exa_search("q")
    mock_set.assert_called_once()


@pytest.mark.asyncio
@patch("src.tools.providers.EXA_API_KEY", "k")
@patch("src.tools.providers.EXA_CACHE_TTL_HOURS", 24)
@patch("src.utils.cache.set_cached_search")
@patch("src.utils.cache.get_cached_search", return_value=None)
@patch("src.tools.providers._request", new_callable=AsyncMock)
async def test_exa_does_not_cache_empty_results(mock_req, mock_get, mock_set):
    """Caching an empty/error-shaped response would poison the key for hours."""
    mock_req.return_value = MagicMock(json=lambda: {"results": []})
    await exa_search("q")
    mock_set.assert_not_called()


@pytest.mark.asyncio
@patch("src.tools.providers.EXA_API_KEY", "k")
@patch("src.tools.providers.EXA_CACHE_TTL_HOURS", 24)
@patch("src.utils.cache.set_cached_search")
@patch("src.utils.cache.get_cached_search", return_value="{not valid json")
@patch("src.tools.providers._request", new_callable=AsyncMock)
async def test_exa_corrupt_cache_entry_falls_through_to_network(mock_req, mock_get, mock_set):
    """A bad cache entry must not block the query."""
    mock_req.return_value = MagicMock(json=lambda: {"results": [{"url": "https://fresh.example/a"}]})
    results = await exa_search("q")
    assert [r.url for r in results] == ["https://fresh.example/a"]
    mock_req.assert_awaited_once()


# ── EDGAR primary_doc enrichment ─────────────────────────────────────────────

def test_edgar_primary_doc_url():
    url = _edgar_primary_doc_url("0002111705-26-000001", "0002111705")
    assert url == (
        "https://www.sec.gov/Archives/edgar/data/2111705/"
        "000211170526000001/primary_doc.xml"
    )


def test_parse_form_d_xml_extracts_money_fields():
    xml = """<edgarSubmission>
      <entityName>HII SpaceX Series II, a Series of HII SpaceX, LLC</entityName>
      <industryGroupType>Pooled Investment Fund</industryGroupType>
      <totalOfferingAmount>15888540</totalOfferingAmount>
      <totalAmountSold>15888540</totalAmountSold>
      <totalRemaining>0</totalRemaining>
      <minimumInvestmentAccepted>25000</minimumInvestmentAccepted>
    </edgarSubmission>"""
    f = _parse_form_d_xml(xml)
    assert f["entityName"].startswith("HII SpaceX Series II")
    assert f["totalOfferingAmount"] == "15888540"
    assert f["totalAmountSold"] == "15888540"


def test_parse_form_d_xml_handles_garbage():
    assert _parse_form_d_xml("") == {}
    assert _parse_form_d_xml("<not-xml") == {}


def test_form_d_text_renders_amounts_and_the_valuation_caveat():
    text = _form_d_text(
        {"entityName": "Acme Fund LP", "totalOfferingAmount": "15888540"}, "D", "2026-03-02"
    )
    assert "Acme Fund LP" in text
    assert "$15,888,540" in text
    # The model must not read an offering amount as a valuation.
    assert "does not state the issuer's valuation" in text


def test_form_d_text_does_not_dollarise_non_numeric_amounts():
    """Form D permits 'Indefinite' as an offering amount."""
    text = _form_d_text({"entityName": "X", "totalOfferingAmount": "Indefinite"}, "D", "")
    assert "Indefinite" in text
    assert "$Indefinite" not in text


def test_form_d_text_empty_when_no_fields():
    assert _form_d_text({}, "D", "2026-01-01") == ""


@pytest.mark.asyncio
@patch("src.tools.providers._request", new_callable=AsyncMock)
async def test_enrich_edgar_sets_text_and_skips_scraping(mock_req):
    mock_req.return_value = MagicMock(
        text="<x><entityName>Acme LP</entityName><totalAmountSold>500</totalAmountSold></x>"
    )
    r = SearchResult(url="https://sec.gov/x", source="sec_edgar",
                     extra={"accession": "0001-26-000001", "ciks": ["0000001"], "form": "D"})
    await _enrich_edgar_filings([r])
    assert "Acme LP" in r.text
    assert r.to_candidate()["needs_scrape"] is False


@pytest.mark.asyncio
@patch("src.tools.providers._request", new_callable=AsyncMock, return_value=None)
async def test_enrich_edgar_missing_xml_leaves_result_scrapable(mock_req):
    """Not every form has a primary_doc.xml — that must degrade, not raise."""
    r = SearchResult(url="https://sec.gov/x", snippet="a filing", source="sec_edgar",
                     extra={"accession": "0001-26-000001", "ciks": ["0000001"], "form": "10-K"})
    await _enrich_edgar_filings([r])
    assert r.text == ""
    assert r.to_candidate()["needs_scrape"] is True


# ── CIK resolution / SPV detection ───────────────────────────────────────────

def test_is_pooled_vehicle():
    assert _is_pooled_vehicle({"industryGroupType": "Pooled Investment Fund"}) is True
    assert _is_pooled_vehicle({"industryGroupType": "Other Technology"}) is False
    assert _is_pooled_vehicle({}) is False


def test_form_d_text_flags_pooled_vehicles():
    """An SPV's raise must never be reportable as the target company's round."""
    text = _form_d_text({
        "entityName": "Founders Fund SpaceX Fund, LP",
        "industryGroupType": "Pooled Investment Fund",
        "totalAmountSold": "3481500",
    }, "D", "2009-05-15")
    assert "pooled investment vehicle" in text
    assert "must NOT be reported as a funding round" in text


def test_form_d_text_does_not_flag_operating_companies():
    text = _form_d_text({
        "entityName": "SPACE EXPLORATION TECHNOLOGIES CORP",
        "industryGroupType": "Other Technology",
        "totalAmountSold": "250000000",
    }, "D", "2022-08-05")
    assert "pooled investment vehicle" not in text


@pytest.mark.asyncio
@patch("src.tools.providers._cik_index", {"spcx": "0001181412"})
async def test_resolve_cik_matches_ticker_exactly():
    assert await resolve_cik("SPCX") == "0001181412"
    assert await resolve_cik("spcx") == "0001181412"


@pytest.mark.asyncio
@patch("src.tools.providers._cik_index", {"spcx": "0001181412"})
async def test_resolve_cik_refuses_ambiguous_short_names():
    """"SpaceX" appears in dozens of SPV names — resolving it would pick wrong."""
    assert await resolve_cik("SpaceX") == ""
    assert await resolve_cik("") == ""


@pytest.mark.asyncio
@patch("src.tools.providers._cik_index", {"spcx": "0001181412"})
@patch("src.tools.providers._enrich_edgar_filings", new_callable=AsyncMock)
@patch("src.tools.providers._request", new_callable=AsyncMock)
async def test_edgar_scopes_by_cik_not_full_text(mock_req, mock_enrich):
    mock_req.return_value = MagicMock(json=lambda: {"hits": {"hits": []}})
    await edgar_search("SpaceX funding", cik="SPCX", forms="D")
    params = mock_req.await_args.kwargs["params"]
    assert params["ciks"] == "0001181412"
    assert "q" not in params          # issuer-scoped, not a name search
    assert params["forms"] == "D"


@pytest.mark.asyncio
@patch("src.tools.providers._cik_index", {"spcx": "0001181412"})
@patch("src.tools.providers._enrich_edgar_filings", new_callable=AsyncMock)
@patch("src.tools.providers._request", new_callable=AsyncMock)
async def test_edgar_falls_back_to_full_text_when_cik_unknown(mock_req, mock_enrich):
    mock_req.return_value = MagicMock(json=lambda: {"hits": {"hits": []}})
    await edgar_search("some private startup")
    params = mock_req.await_args.kwargs["params"]
    assert "ciks" not in params
    assert params["q"] == '"some private startup"'


# ── Europe PMC ───────────────────────────────────────────────────────────────

def test_parse_europe_pmc_builds_article_url_and_metadata():
    data = {"resultList": {"result": [{
        "source": "MED", "id": "31401015",
        "title": "GLP-1 Receptor Agonist Cardiovascular Outcomes Trials.",
        "abstractText": "<h4>Background</h4>The latest recommendations...",
        "pubYear": "2019", "citedByCount": 43, "authorString": "Smith J, Doe A.",
        "journalInfo": {"journal": {"title": "Trends in Endocrinology"}},
    }]}}
    r = _parse_europe_pmc(data)[0]
    assert r.url == "https://europepmc.org/article/MED/31401015"
    assert r.as_of_date == "2019"
    assert r.authors == ["Smith J", "Doe A."]
    assert r.extra["journal"] == "Trends in Endocrinology"
    assert "<h4>" not in r.snippet          # tags stripped for the extractor
    assert r.credibility_hint == 0.95
    assert r.to_candidate()["needs_scrape"] is False   # abstract is usable content


def test_parse_europe_pmc_grades_preprints_below_published_work():
    data = {"resultList": {"result": [{
        "source": "PPR", "id": "PPR1308671", "title": "A preprint",
        "abstractText": "Not yet peer reviewed.", "pubYear": "2026",
    }]}}
    r = _parse_europe_pmc(data)[0]
    assert r.extra["is_preprint"] is True
    assert r.credibility_hint == 0.75          # below the 0.95 for published
    assert r.source_type_hint == "Preprint/Biomedical"


def test_parse_europe_pmc_skips_records_without_identifiers():
    assert _parse_europe_pmc({"resultList": {"result": [{"title": "no ids"}]}}) == []


def test_parse_europe_pmc_handles_empty_and_garbage():
    assert _parse_europe_pmc({}) == []
    assert _parse_europe_pmc({"resultList": None}) == []


@pytest.mark.asyncio
async def test_europe_pmc_empty_query_returns_empty():
    assert await europe_pmc_search("   ") == []


@pytest.mark.asyncio
@patch("src.tools.providers._request", new_callable=AsyncMock, return_value=None)
async def test_europe_pmc_degrades_on_network_failure(mock_req):
    assert await europe_pmc_search("anything") == []


# ── Semantic Scholar quota discipline ────────────────────────────────────────

@pytest.mark.asyncio
@patch("src.tools.providers.SEMANTIC_SCHOLAR_CACHE_TTL_HOURS", 24)
@patch("src.utils.cache.set_cached_search")
@patch("src.utils.cache.get_cached_search")
@patch("src.tools.providers._request", new_callable=AsyncMock)
async def test_semantic_scholar_cache_hit_sends_no_request(mock_req, mock_get, mock_set):
    """A repeated sub-query must be served from cache, not re-spend quota."""
    mock_get.return_value = '{"data": [{"title": "Cached paper", "url": "http://x"}]}'
    results = await semantic_scholar_search("some query")
    assert [r.title for r in results] == ["Cached paper"]
    mock_req.assert_not_awaited()
    mock_set.assert_not_called()


@pytest.mark.asyncio
@patch("src.tools.providers.SEMANTIC_SCHOLAR_CACHE_TTL_HOURS", 24)
@patch("src.utils.cache.set_cached_search")
@patch("src.utils.cache.get_cached_search", return_value=None)
@patch("src.tools.providers._request", new_callable=AsyncMock)
async def test_semantic_scholar_caches_successful_response(mock_req, mock_get, mock_set):
    mock_req.return_value = MagicMock(json=lambda: {"data": [{"title": "P", "url": "http://x"}]})
    await semantic_scholar_search("q")
    mock_set.assert_called_once()


@pytest.mark.asyncio
@patch("src.tools.providers.SEMANTIC_SCHOLAR_CACHE_TTL_HOURS", 24)
@patch("src.utils.cache.set_cached_search")
@patch("src.utils.cache.get_cached_search", return_value=None)
@patch("src.tools.providers._request", new_callable=AsyncMock)
async def test_semantic_scholar_does_not_cache_a_throttled_response(mock_req, mock_get, mock_set):
    """A 429 yields no results; caching that would suppress retries for hours."""
    mock_req.return_value = None
    assert await semantic_scholar_search("q") == []
    mock_set.assert_not_called()


@pytest.mark.asyncio
@patch("src.tools.providers.SEMANTIC_SCHOLAR_RPS", 4.0)
async def test_semantic_scholar_throttle_serialises_concurrent_callers():
    """The Scout gathers sub-queries concurrently; the 1 RPS allowance is held
    client-side rather than relying on the server to reject bursts."""
    import asyncio, time
    from src.tools.providers import _s2_throttle
    start = time.perf_counter()
    await asyncio.gather(*[_s2_throttle() for _ in range(4)])
    elapsed = time.perf_counter() - start
    assert elapsed >= 0.7, f"4 calls at 4 RPS should take ~0.75s, took {elapsed:.2f}s"
