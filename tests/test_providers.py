"""
Project Argus - Search provider parser tests (pure, no network)

Exercises the PURE parse functions that turn raw API payloads into
normalized SearchResults, plus the candidate shaping logic.
"""

from src.tools.providers import (
    SearchResult,
    _parse_semantic_scholar,
    _parse_arxiv,
    _parse_crossref,
    _parse_brave,
    _ddgs_to_results,
    _clean_text,
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
