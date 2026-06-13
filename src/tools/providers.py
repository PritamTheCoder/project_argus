"""
Project Argus - Search Provider Backends (Phase 2)

Real, usable search backends behind a single normalized result type
(`SearchResult`). Each backend has:

  - an async network function (httpx, timeouts, retries, graceful empty on error)
  - a PURE parse function (`_parse_*`) that turns the raw API payload into
    `SearchResult`s and is unit-tested without any network.

Backends:
  - semantic_scholar_search : 200M+ papers, abstracts + metadata (free, optional key)
  - arxiv_search            : preprints, Atom XML (free, no key)
  - crossref_search         : DOI metadata across publishers (free, polite pool)
  - web_search              : Brave Search API if BRAVE_API_KEY set, else DuckDuckGo

Academic backends return the abstract as `content` (no scraping needed — the
whole point of using the APIs). Web results carry only a snippet and are flagged
`needs_scrape=True` so the Scout fetches full text.
"""

from __future__ import annotations

import asyncio
import logging
import re
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional

import httpx
from pydantic import BaseModel, Field

from src.config import (
    BRAVE_API_KEY,
    SEMANTIC_SCHOLAR_API_KEY,
    CROSSREF_MAILTO,
    TOOL_HTTP_TIMEOUT,
    ACADEMIC_MAX_RESULTS,
)

logger = logging.getLogger(__name__)

_USER_AGENT = "ProjectArgus/1.0 (research agent; +https://project-argus.local)"


class SearchResult(BaseModel):
    """Normalized search hit from any backend."""
    title: str = ""
    url: str = ""
    snippet: str = ""            # abstract (academic) or result description (web)
    source: str = "web"         # backend id: semantic_scholar | arxiv | crossref | web
    as_of_date: str = ""        # year or ISO date if known
    authors: List[str] = Field(default_factory=list)
    extra: Dict[str, Any] = Field(default_factory=dict)

    # Credibility hint (None → let the Scout's domain scorer decide).
    credibility_hint: Optional[float] = None
    source_type_hint: Optional[str] = None

    def to_candidate(self) -> Dict[str, Any]:
        """Shape consumed by the Scout acquisition pipeline."""
        is_academic = self.source in ("semantic_scholar", "arxiv", "crossref")
        # Academic backends already give us the abstract → no scrape needed.
        content = self.snippet if is_academic else ""
        return {
            "url": self.url,
            "title": self.title,
            "snippet": self.snippet,
            "content": content,
            "source": self.source,
            "as_of_date": self.as_of_date,
            "needs_scrape": not (is_academic and bool(content.strip())),
            "credibility_hint": self.credibility_hint,
            "source_type_hint": self.source_type_hint,
            "authors": self.authors,
        }


# ── Shared HTTP helpers ──────────────────────────────────────────────────────

async def _request(method: str, url: str, *, retries: int = 2, **kwargs) -> Optional[httpx.Response]:
    """HTTP request with timeout + light retry. Returns None on persistent failure."""
    kwargs.setdefault("timeout", TOOL_HTTP_TIMEOUT)
    headers = kwargs.pop("headers", {}) or {}
    headers.setdefault("User-Agent", _USER_AGENT)

    last_err: Optional[Exception] = None
    for attempt in range(retries + 1):
        try:
            async with httpx.AsyncClient(follow_redirects=True) as client:
                resp = await client.request(method, url, headers=headers, **kwargs)
            if resp.status_code == 429:
                # Rate limited — brief backoff then retry.
                await asyncio.sleep(1.0 * (attempt + 1))
                last_err = httpx.HTTPStatusError("429", request=resp.request, response=resp)
                continue
            resp.raise_for_status()
            return resp
        except Exception as e:  # noqa: BLE001 - we want to swallow and degrade
            last_err = e
            if attempt < retries:
                await asyncio.sleep(0.5 * (attempt + 1))
    logger.warning("HTTP %s %s failed after %d attempts: %s", method, url, retries + 1, last_err)
    return None


def _clean_text(s: Optional[str]) -> str:
    if not s:
        return ""
    # Strip JATS/HTML tags (common in Crossref abstracts) and collapse whitespace.
    s = re.sub(r"<[^>]+>", " ", s)
    return re.sub(r"\s+", " ", s).strip()


# ── Semantic Scholar ─────────────────────────────────────────────────────────

_SEMANTIC_SCHOLAR_URL = "https://api.semanticscholar.org/graph/v1/paper/search"


def _parse_semantic_scholar(data: Dict[str, Any]) -> List[SearchResult]:
    out: List[SearchResult] = []
    for p in (data or {}).get("data", []) or []:
        if not p:
            continue
        url = p.get("url") or ""
        oa = p.get("openAccessPdf") or {}
        if oa.get("url"):
            url = oa["url"]
        out.append(SearchResult(
            title=p.get("title") or "",
            url=url,
            snippet=_clean_text(p.get("abstract")),
            source="semantic_scholar",
            as_of_date=str(p.get("year") or ""),
            authors=[a.get("name", "") for a in (p.get("authors") or []) if a.get("name")],
            extra={"citationCount": p.get("citationCount"), "externalIds": p.get("externalIds")},
            credibility_hint=0.9,
            source_type_hint="Academic/Scientific",
        ))
    return out


async def semantic_scholar_search(query: str, limit: int = ACADEMIC_MAX_RESULTS) -> List[SearchResult]:
    if not query.strip():
        return []
    params = {
        "query": query,
        "limit": max(1, min(limit, 100)),
        "fields": "title,abstract,year,url,authors,citationCount,externalIds,openAccessPdf",
    }
    headers = {}
    if SEMANTIC_SCHOLAR_API_KEY:
        headers["x-api-key"] = SEMANTIC_SCHOLAR_API_KEY
    resp = await _request("GET", _SEMANTIC_SCHOLAR_URL, params=params, headers=headers)
    if resp is None:
        return []
    try:
        return _parse_semantic_scholar(resp.json())
    except Exception as e:  # noqa: BLE001
        logger.warning("Semantic Scholar parse failed: %s", e)
        return []


# ── arXiv ────────────────────────────────────────────────────────────────────

_ARXIV_URL = "http://export.arxiv.org/api/query"
_ATOM_NS = {"atom": "http://www.w3.org/2005/Atom"}


def _parse_arxiv(xml_text: str) -> List[SearchResult]:
    out: List[SearchResult] = []
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as e:
        logger.warning("arXiv XML parse failed: %s", e)
        return []
    for entry in root.findall("atom:entry", _ATOM_NS):
        title = (entry.findtext("atom:title", default="", namespaces=_ATOM_NS) or "").strip()
        summary = (entry.findtext("atom:summary", default="", namespaces=_ATOM_NS) or "").strip()
        url = (entry.findtext("atom:id", default="", namespaces=_ATOM_NS) or "").strip()
        published = (entry.findtext("atom:published", default="", namespaces=_ATOM_NS) or "").strip()
        authors = [
            (a.findtext("atom:name", default="", namespaces=_ATOM_NS) or "").strip()
            for a in entry.findall("atom:author", _ATOM_NS)
        ]
        out.append(SearchResult(
            title=_clean_text(title),
            url=url,
            snippet=_clean_text(summary),
            source="arxiv",
            as_of_date=published[:10] if published else "",
            authors=[a for a in authors if a],
            credibility_hint=0.85,
            source_type_hint="Academic/Scientific",
        ))
    return out


async def arxiv_search(query: str, limit: int = ACADEMIC_MAX_RESULTS) -> List[SearchResult]:
    if not query.strip():
        return []
    params = {
        "search_query": f"all:{query}",
        "start": 0,
        "max_results": max(1, min(limit, 50)),
        "sortBy": "relevance",
    }
    resp = await _request("GET", _ARXIV_URL, params=params)
    if resp is None:
        return []
    return _parse_arxiv(resp.text)


# ── Crossref ─────────────────────────────────────────────────────────────────

_CROSSREF_URL = "https://api.crossref.org/works"


def _parse_crossref(data: Dict[str, Any]) -> List[SearchResult]:
    out: List[SearchResult] = []
    for item in (data or {}).get("message", {}).get("items", []) or []:
        if not item:
            continue
        title_list = item.get("title") or []
        title = _clean_text(title_list[0]) if title_list else ""
        doi = item.get("DOI") or ""
        url = item.get("URL") or (f"https://doi.org/{doi}" if doi else "")
        abstract = _clean_text(item.get("abstract"))
        container = item.get("container-title") or []
        venue = container[0] if container else ""
        # Year from issued date-parts.
        year = ""
        dp = (item.get("issued") or {}).get("date-parts") or []
        if dp and dp[0]:
            year = str(dp[0][0])
        authors = [
            " ".join(x for x in [a.get("given", ""), a.get("family", "")] if x).strip()
            for a in (item.get("author") or [])
        ]
        snippet = abstract or (f"{title} — {venue}" if venue else title)
        out.append(SearchResult(
            title=title,
            url=url,
            snippet=snippet,
            source="crossref",
            as_of_date=year,
            authors=[a for a in authors if a],
            extra={"venue": venue, "doi": doi},
            credibility_hint=0.9,
            source_type_hint="Academic/Scientific",
        ))
    return out


async def crossref_search(query: str, limit: int = ACADEMIC_MAX_RESULTS) -> List[SearchResult]:
    if not query.strip():
        return []
    params = {
        "query": query,
        "rows": max(1, min(limit, 50)),
        "select": "title,abstract,DOI,URL,issued,author,container-title",
        "mailto": CROSSREF_MAILTO,
    }
    resp = await _request("GET", _CROSSREF_URL, params=params)
    if resp is None:
        return []
    try:
        return _parse_crossref(resp.json())
    except Exception as e:  # noqa: BLE001
        logger.warning("Crossref parse failed: %s", e)
        return []


# ── Web (Brave → DuckDuckGo fallback) ────────────────────────────────────────

_BRAVE_URL = "https://api.search.brave.com/res/v1/web/search"


def _parse_brave(data: Dict[str, Any]) -> List[SearchResult]:
    out: List[SearchResult] = []
    for r in (data or {}).get("web", {}).get("results", []) or []:
        if not r or not r.get("url"):
            continue
        out.append(SearchResult(
            title=_clean_text(r.get("title")),
            url=r.get("url"),
            snippet=_clean_text(r.get("description")),
            source="web",
            as_of_date=(r.get("page_age") or "")[:10],
        ))
    return out


async def _brave_web_search(query: str, limit: int) -> List[SearchResult]:
    params = {"q": query, "count": max(1, min(limit, 20))}
    headers = {"Accept": "application/json", "X-Subscription-Token": BRAVE_API_KEY}
    resp = await _request("GET", _BRAVE_URL, params=params, headers=headers)
    if resp is None:
        return []
    try:
        return _parse_brave(resp.json())
    except Exception as e:  # noqa: BLE001
        logger.warning("Brave parse failed: %s", e)
        return []


def _ddgs_to_results(raw: List[Dict[str, Any]]) -> List[SearchResult]:
    """Pure conversion of DDGS().text() output to SearchResult list."""
    out: List[SearchResult] = []
    for r in raw or []:
        url = r.get("href") or r.get("url") or ""
        if not url:
            continue
        out.append(SearchResult(
            title=_clean_text(r.get("title")),
            url=url,
            snippet=_clean_text(r.get("body")),
            source="web",
        ))
    return out


async def _ddg_web_search(query: str, limit: int) -> List[SearchResult]:
    # DDGS is sync; run it off the event loop.
    def _run() -> List[Dict[str, Any]]:
        from ddgs import DDGS
        with DDGS() as ddgs:
            return list(ddgs.text(query, max_results=limit * 2))
    try:
        raw = await asyncio.to_thread(_run)
    except Exception as e:  # noqa: BLE001
        logger.warning("DuckDuckGo search failed: %s", e)
        return []
    return _ddgs_to_results(raw)


async def web_search(query: str, limit: int = 8) -> List[SearchResult]:
    """Brave Search if a key is configured, else DuckDuckGo. Junk domains filtered."""
    if not query.strip():
        return []
    results = await _brave_web_search(query, limit) if BRAVE_API_KEY else await _ddg_web_search(query, limit)

    # Filter junk domains (reuse the existing policy from the search tool).
    from src.tools.search import _is_junk_domain
    filtered = [r for r in results if r.url and not _is_junk_domain(r.url)]
    return filtered[:limit]
