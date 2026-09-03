"""
Search provider backends behind a single normalized result type
(`SearchResult`). Each backend has:

  - an async network function (httpx, timeouts, retries, graceful empty on error)
  - a PURE parse function (`_parse_*`) that turns the raw API payload into
    `SearchResult`s and is unit-tested without any network.

Backends:
  - semantic_scholar_search : 200M+ papers, abstracts + metadata (free, optional key)
  - arxiv_search            : preprints, Atom XML (free, no key)
  - crossref_search         : DOI metadata across publishers (free, polite pool)
  - europe_pmc_search       : PubMed/MEDLINE + PMC + preprints (free, no key)
  - edgar_search            : SEC filings, US regulatory primary sources (free, no key)
  - exa_search              : neural search with category filters (metered, cached)
  - web_search              : ladder over Exa → Brave → DuckDuckGo

Backends that already return usable content (academic abstracts, Exa text) set
`needs_scrape=False` so the Scout skips the browser. Results carrying only a
snippet are flagged `needs_scrape=True` and get fetched.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional

import httpx
from pydantic import BaseModel, Field

from src.config import (
    BRAVE_API_KEY,
    EXA_API_KEY,
    EXA_TEXT_CHARS,
    EXA_CONTENT_MODE,
    EXA_SEARCH_TYPE,
    EXA_MAX_RESULTS,
    EXA_CACHE_TTL_HOURS,
    EDGAR_USER_AGENT,
    EDGAR_MAX_RESULTS,
    EDGAR_ENRICH_CONCURRENCY,
    SEMANTIC_SCHOLAR_API_KEY,
    SEMANTIC_SCHOLAR_RPS,
    SEMANTIC_SCHOLAR_CACHE_TTL_HOURS,
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
    text: str = ""              # full page text, when a backend supplies it (e.g. Exa)
    source: str = "web"         # backend id: semantic_scholar | arxiv | crossref | exa | sec_edgar | web
    as_of_date: str = ""        # year or ISO date if known
    authors: List[str] = Field(default_factory=list)
    extra: Dict[str, Any] = Field(default_factory=dict)

    # Credibility hint (None → let the Scout's domain scorer decide).
    credibility_hint: Optional[float] = None
    source_type_hint: Optional[str] = None

    def to_candidate(self) -> Dict[str, Any]:
        """Shape consumed by the Scout acquisition pipeline."""
        is_academic = self.source in ("semantic_scholar", "arxiv", "crossref", "europe_pmc")
        # Usable content is either full text a backend already fetched, or an
        # academic abstract. Anything else still has to be scraped.
        content = self.text or (self.snippet if is_academic else "")
        return {
            "url": self.url,
            "title": self.title,
            "snippet": self.snippet,
            "content": content,
            "source": self.source,
            "as_of_date": self.as_of_date,
            "needs_scrape": not bool(content.strip()),
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
        except Exception as e:  # swallow and degrade to None on persistent failure
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


# Semantic Scholar's introductory authenticated limit is 1 request/second, and
# the Scout issues sub-queries concurrently — so requests are serialised through
# a minimum-interval gate rather than relying on the server to reject bursts.
_S2_LOCK = asyncio.Lock()
_s2_last_request = 0.0


async def _s2_throttle() -> None:
    """Hold each Semantic Scholar request to at most SEMANTIC_SCHOLAR_RPS."""
    global _s2_last_request
    interval = 1.0 / max(SEMANTIC_SCHOLAR_RPS, 0.1)
    async with _S2_LOCK:
        wait = interval - (asyncio.get_event_loop().time() - _s2_last_request)
        if wait > 0:
            await asyncio.sleep(wait)
        _s2_last_request = asyncio.get_event_loop().time()


async def semantic_scholar_search(query: str, limit: int = ACADEMIC_MAX_RESULTS) -> List[SearchResult]:
    """Search Semantic Scholar. Client-side throttled and cached.

    A research run re-asks overlapping sub-queries across loop iterations, so
    caching removes repeat requests entirely rather than re-spending quota.
    """
    if not query.strip():
        return []
    params = {
        "query": query,
        "limit": max(1, min(limit, 100)),
        "fields": "title,abstract,year,url,authors,citationCount,externalIds,openAccessPdf",
    }

    from src.utils.cache import get_cached_search, set_cached_search
    cache_key = "s2:" + hashlib.sha256(
        json.dumps(params, sort_keys=True).encode("utf-8")
    ).hexdigest()

    cached = get_cached_search(cache_key, SEMANTIC_SCHOLAR_CACHE_TTL_HOURS)
    if cached is not None:
        try:
            logger.info("Semantic Scholar cache hit for %r (no request sent).", query)
            return _parse_semantic_scholar(json.loads(cached))
        except Exception as e:  # noqa: BLE001 — a bad entry must not block the query
            logger.warning("Semantic Scholar cache entry unusable (%s); re-querying.", e)

    headers = {}
    if SEMANTIC_SCHOLAR_API_KEY:
        headers["x-api-key"] = SEMANTIC_SCHOLAR_API_KEY

    await _s2_throttle()
    resp = await _request("GET", _SEMANTIC_SCHOLAR_URL, params=params, headers=headers)
    if resp is None:
        return []
    try:
        data = resp.json()
    except Exception as e:  # noqa: BLE001
        logger.warning("Semantic Scholar parse failed: %s", e)
        return []

    results = _parse_semantic_scholar(data)
    if results:  # never cache an empty/error-shaped response
        set_cached_search(cache_key, json.dumps(data))
    return results


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


# ── Europe PMC (biomedical / life sciences) ──────────────────────────────────

_EUROPE_PMC_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
_EUROPE_PMC_ARTICLE = "https://europepmc.org/article"


def _parse_europe_pmc(data: Dict[str, Any]) -> List[SearchResult]:
    out: List[SearchResult] = []
    for r in ((data or {}).get("resultList") or {}).get("result", []) or []:
        src, pid = r.get("source"), r.get("id")
        if not src or not pid:
            continue
        journal = ((r.get("journalInfo") or {}).get("journal") or {}).get("title", "")
        abstract = _clean_text(r.get("abstractText"))
        # Preprints (source PPR) are not peer-reviewed; grade them below the
        # published literature so the Writer hedges them accordingly.
        is_preprint = src == "PPR"
        out.append(SearchResult(
            title=_clean_text(r.get("title")),
            url=f"{_EUROPE_PMC_ARTICLE}/{src}/{pid}",
            snippet=abstract,
            source="europe_pmc",
            as_of_date=str(r.get("pubYear") or ""),
            authors=[a.strip() for a in (r.get("authorString") or "").split(",") if a.strip()],
            extra={"journal": journal, "cited_by": r.get("citedByCount"),
                   "is_preprint": is_preprint, "doi": r.get("doi")},
            credibility_hint=0.75 if is_preprint else 0.95,
            source_type_hint="Preprint/Biomedical" if is_preprint else "Academic/Biomedical",
        ))
    return out


async def europe_pmc_search(query: str, limit: int = ACADEMIC_MAX_RESULTS) -> List[SearchResult]:
    """Search Europe PMC — PubMed/MEDLINE, PMC full text, and preprints.

    Free, no key, no advertised quota. This is the correct corpus for clinical
    and life-sciences questions; arXiv does not cover them, and falling through
    to it produces authoritative-looking sources on the wrong subject.
    Returns [] on any failure so the caller keeps whatever else it gathered.
    """
    if not query.strip():
        return []
    params = {
        "query": query,
        "format": "json",
        "pageSize": max(1, min(limit, 100)),
        "resultType": "core",   # includes abstracts, so results need no scraping
    }
    resp = await _request("GET", _EUROPE_PMC_URL, params=params)
    if resp is None:
        return []
    try:
        return _parse_europe_pmc(resp.json())
    except Exception as e:  # noqa: BLE001
        logger.warning("Europe PMC parse failed: %s", e)
        return []


# ── Exa (neural search) ──────────────────────────────────────────────────────

_EXA_URL = "https://api.exa.ai/search"

# Categories Exa accepts, per the canonical reference. Anything else is dropped
# rather than sent, so a bad value from the model degrades to an ordinary search
# instead of a 400.
# Note: `company` and `people` reject excludeDomains and date filters — don't
# combine them if those are ever added here.
EXA_CATEGORIES = {
    "company", "people", "publication", "news", "personal site", "financial report",
}


def _parse_exa(data: Dict[str, Any]) -> List[SearchResult]:
    out: List[SearchResult] = []
    for r in (data or {}).get("results", []) or []:
        if not r or not r.get("url"):
            continue
        # `highlights` mode returns a list of query-relevant excerpts; join them
        # so either content mode lands in the same `text` field downstream.
        text = (r.get("text") or "").strip()
        if not text and isinstance(r.get("highlights"), list):
            text = "\n\n".join(h for h in r["highlights"] if isinstance(h, str)).strip()
        out.append(SearchResult(
            title=_clean_text(r.get("title")),
            url=r["url"],
            # Prefer Exa's own summary; otherwise lead with the content it returned.
            snippet=_clean_text(r.get("summary")) or _clean_text(text[:400]),
            text=text,
            source="exa",
            as_of_date=(r.get("publishedDate") or "")[:10],
            authors=[r["author"]] if r.get("author") else [],
            extra={"score": r.get("score")},
        ))
    return out


def _exa_contents_payload() -> Dict[str, Any]:
    """Content block for a /search request.

    `text` is the default because this pipeline extracts facts and then grounds
    verbatim quotes against the source excerpt — that needs continuous prose, not
    fragments. `highlights` is cheaper per result and available via
    EXA_CONTENT_MODE for token-constrained runs.
    """
    if EXA_CONTENT_MODE == "highlights":
        return {"highlights": True}
    return {"text": {"maxCharacters": EXA_TEXT_CHARS}}


async def exa_search(query: str, limit: int = 8, category: Optional[str] = None) -> List[SearchResult]:
    """Exa neural search, optionally biased to a `category` (see EXA_CATEGORIES).

    Exa is metered ($7/1k searches + $1/1k pages of contents), so this caps
    results at EXA_MAX_RESULTS and caches responses for EXA_CACHE_TTL_HOURS —
    re-running a query costs nothing on a cache hit.

    Returns [] when no key is set or on any failure, so callers fall through to
    the next backend rather than losing the query.
    """
    if not EXA_API_KEY or not query.strip():
        return []

    num_results = max(1, min(limit, EXA_MAX_RESULTS))
    payload: Dict[str, Any] = {
        "query": query,
        "type": EXA_SEARCH_TYPE,
        "numResults": num_results,
        "contents": _exa_contents_payload(),
    }
    if category and category in EXA_CATEGORIES:
        payload["category"] = category

    from src.utils.cache import get_cached_search, set_cached_search
    cache_key = "exa:" + hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()

    cached = get_cached_search(cache_key, EXA_CACHE_TTL_HOURS)
    if cached is not None:
        try:
            logger.info("Exa cache hit for %r (no request billed).", query)
            return _parse_exa(json.loads(cached))
        except Exception as e:  # noqa: BLE001 — a bad cache entry must not block the query
            logger.warning("Exa cache entry unusable (%s); re-querying.", e)

    resp = await _request(
        "POST", _EXA_URL, json=payload,
        headers={"x-api-key": EXA_API_KEY, "Content-Type": "application/json"},
    )
    if resp is None:
        return []
    try:
        data = resp.json()
    except Exception as e:  # noqa: BLE001
        logger.warning("Exa returned unparseable JSON: %s", e)
        return []

    results = _parse_exa(data)
    if results:  # don't cache empty/error-shaped responses
        set_cached_search(cache_key, json.dumps(data))
    return results


# ── SEC EDGAR (US regulatory primary sources) ────────────────────────────────

_EDGAR_SEARCH_URL = "https://efts.sec.gov/LATEST/search-index"
_EDGAR_ARCHIVE = "https://www.sec.gov/Archives/edgar/data"


def _edgar_filing_url(accession: str, cik: str) -> str:
    """Filing index page from its accession number and CIK.

    EDGAR archive paths strip the CIK's leading zeros and the accession's
    dashes: 0001464202 + 0001464202-09-000001 →
    /data/1464202/000146420209000001/0001464202-09-000001-index.htm
    """
    if not accession or not cik:
        return ""
    return f"{_EDGAR_ARCHIVE}/{cik.lstrip('0')}/{accession.replace('-', '')}/{accession}-index.htm"


def _edgar_primary_doc_url(accession: str, cik: str) -> str:
    """The filing's structured primary document, alongside its index page."""
    if not accession or not cik:
        return ""
    return f"{_EDGAR_ARCHIVE}/{cik.lstrip('0')}/{accession.replace('-', '')}/primary_doc.xml"


# Form D fields carrying the substance: who raised, how much, and how much sold.
_FORM_D_FIELDS = (
    "entityName", "industryGroupType", "totalOfferingAmount",
    "totalAmountSold", "totalRemaining", "minimumInvestmentAccepted",
)


def _parse_form_d_xml(xml_text: str) -> Dict[str, str]:
    """Extract the money fields from a Form D primary_doc.xml. Pure."""
    out: Dict[str, str] = {}
    for field in _FORM_D_FIELDS:
        m = re.search(rf"<{field}>([^<]*)</{field}>", xml_text or "")
        if m and m.group(1).strip():
            out[field] = m.group(1).strip()
    return out


def _is_pooled_vehicle(fields: Dict[str, str]) -> bool:
    """True when the filer is an investment vehicle rather than an operating company.

    EDGAR full-text search for a company name mostly returns SPVs and feeder
    funds formed to buy into that company. Their offering amounts are the
    vehicle's own raise, not the target's funding round, and the two must not
    be conflated.
    """
    return "pooled investment fund" in fields.get("industryGroupType", "").lower()


def _form_d_text(fields: Dict[str, str], form: str, filed: str) -> str:
    """Render extracted filing fields as prose the Refiner can pull facts from.

    The filing's index page is almost entirely navigation boilerplate, so
    retrieval discards it. This gives the pipeline the actual numbers instead.
    """
    if not fields:
        return ""
    def _amount(v: str) -> str:
        # Form D permits non-numeric amounts such as "Indefinite" — don't
        # render those as a dollar figure.
        return f"${int(v):,}" if v.isdigit() else v

    name = fields.get("entityName", "The filer")
    parts = [f"SEC {form} filing by {name}, filed {filed}." if filed
             else f"SEC {form} filing by {name}."]
    if "industryGroupType" in fields:
        parts.append(f"Industry group: {fields['industryGroupType']}.")
    if "totalOfferingAmount" in fields:
        parts.append(f"Total offering amount: {_amount(fields['totalOfferingAmount'])}.")
    if "totalAmountSold" in fields:
        parts.append(f"Total amount sold: {_amount(fields['totalAmountSold'])}.")
    if "totalRemaining" in fields:
        parts.append(f"Amount remaining to be sold: {_amount(fields['totalRemaining'])}.")
    if "minimumInvestmentAccepted" in fields:
        parts.append(f"Minimum investment accepted: {_amount(fields['minimumInvestmentAccepted'])}.")
    if _is_pooled_vehicle(fields):
        parts.append(
            f"IMPORTANT: {name} is a pooled investment vehicle (an SPV or feeder "
            "fund) raising capital to invest, NOT an operating company. This "
            "amount is the vehicle's own raise and must NOT be reported as a "
            "funding round of the company it invests in."
        )
    parts.append(
        "This is the amount raised in the offering; a Form D does not state the "
        "issuer's valuation."
    )
    return " ".join(parts)


def _parse_edgar(data: Dict[str, Any]) -> List[SearchResult]:
    out: List[SearchResult] = []
    for hit in ((data or {}).get("hits") or {}).get("hits", []) or []:
        src = (hit or {}).get("_source") or {}
        ciks = src.get("ciks") or []
        url = _edgar_filing_url(src.get("adsh") or "", ciks[0] if ciks else "")
        if not url:
            continue
        names = src.get("display_names") or []
        filer = names[0] if names else ""
        form = src.get("form") or src.get("file_type") or "filing"
        filed = src.get("file_date") or ""
        out.append(SearchResult(
            title=f"SEC {form} — {filer}" if filer else f"SEC {form} {src.get('adsh', '')}",
            snippet=f"Official SEC EDGAR {form} filing, filed {filed}."
                    + (f" Filer: {filer}." if filer else ""),
            url=url,
            source="sec_edgar",
            as_of_date=filed,
            extra={"accession": src.get("adsh"), "form": form, "ciks": ciks},
            credibility_hint=0.9,
            source_type_hint="Government/Institutional",
        ))
    return out


_EDGAR_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
_cik_index: Optional[Dict[str, str]] = None


async def resolve_cik(name_or_ticker: str) -> str:
    """Resolve a ticker or registered company name to a zero-padded CIK.

    Matching is deliberately exact (on ticker, or on the full registered title).
    A company's short name commonly appears in unrelated investment-vehicle
    names, so fuzzy matching resolves to the wrong entity more often than the
    right one. Returns "" when unsure; the caller falls back to full-text search.
    """
    global _cik_index
    key = (name_or_ticker or "").strip().lower()
    if not key:
        return ""

    if _cik_index is None:
        from src.utils.cache import get_cached_search, set_cached_search
        raw = get_cached_search("sec:company_tickers", 24 * 7)
        if raw is None:
            resp = await _request("GET", _EDGAR_TICKERS_URL,
                                  headers={"User-Agent": EDGAR_USER_AGENT})
            if resp is None:
                return ""
            raw = resp.text
            set_cached_search("sec:company_tickers", raw)
        try:
            data = json.loads(raw)
        except Exception as e:  # noqa: BLE001
            logger.warning("SEC ticker index unusable: %s", e)
            return ""
        index: Dict[str, str] = {}
        for row in (data or {}).values():
            cik = str(row.get("cik_str", "")).zfill(10)
            for k in (row.get("ticker", ""), row.get("title", "")):
                if k:
                    index[k.strip().lower()] = cik
        _cik_index = index

    return _cik_index.get(key, "")


async def edgar_search(
    query: str,
    limit: int = EDGAR_MAX_RESULTS,
    forms: Optional[str] = None,
    cik: Optional[str] = None,
) -> List[SearchResult]:
    """Search SEC EDGAR filings (free, no key).

    `cik` pins the issuer, which is strongly preferred: full-text search for a
    company name mostly returns SPVs formed to invest in it, not its own
    filings. A ticker or exact registered name is resolved to a CIK
    automatically; otherwise this falls back to full-text search.

    `forms` filters by filing type, e.g. "D" (private placements), "10-K", "8-K".
    Returns [] on any failure so the caller keeps whatever else it gathered.
    """
    if not query.strip() and not cik:
        return []

    resolved = (cik or "").strip()
    if resolved and not resolved.isdigit():
        resolved = await resolve_cik(resolved)
    elif resolved:
        resolved = resolved.zfill(10)
    if not resolved:
        resolved = await resolve_cik(query)

    params: Dict[str, str] = {}
    if resolved:
        # Issuer-scoped: every hit is genuinely this company's filing.
        params["ciks"] = resolved
        logger.info("EDGAR: scoped to CIK %s for %r.", resolved, query)
    else:
        # EDGAR treats a quoted query as a phrase, which is what we want for
        # company names; don't re-quote a query that already carries quoting.
        params["q"] = query if '"' in query else f'"{query}"'
    if forms:
        params["forms"] = forms
    resp = await _request(
        "GET", _EDGAR_SEARCH_URL, params=params,
        headers={"User-Agent": EDGAR_USER_AGENT},
    )
    if resp is None:
        return []
    try:
        results = _parse_edgar(resp.json())[:limit]
    except Exception as e:  # noqa: BLE001
        logger.warning("EDGAR parse failed: %s", e)
        return []

    await _enrich_edgar_filings(results)
    return results


async def _enrich_edgar_filings(results: List[SearchResult]) -> None:
    """Fill each filing's `text` from its primary_doc.xml, in place.

    Without this the pipeline scrapes the filing's index page, which is
    navigation boilerplate — retrieval then ranks it below every blog post and
    the filing is discarded despite being a 0.9-credibility primary source.
    Pulling the structured document instead gives real content AND skips the
    browser entirely.

    Best-effort: a filing whose XML is missing (not every form has one) simply
    keeps its snippet and gets scraped as before. SEC allows 10 req/s, so this
    is bounded.
    """
    sem = asyncio.Semaphore(EDGAR_ENRICH_CONCURRENCY)

    async def _one(r: SearchResult) -> None:
        url = _edgar_primary_doc_url(
            str(r.extra.get("accession") or ""),
            (r.extra.get("ciks") or [""])[0],
        )
        if not url:
            return
        async with sem:
            resp = await _request("GET", url, retries=0,
                                  headers={"User-Agent": EDGAR_USER_AGENT})
        if resp is None:
            return
        text = _form_d_text(_parse_form_d_xml(resp.text), r.extra.get("form", "filing"), r.as_of_date)
        if text:
            r.text = text
            r.snippet = text[:400]

    await asyncio.gather(*(_one(r) for r in results), return_exceptions=True)


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


async def web_search(
    query: str, limit: int = 8, category: Optional[str] = None
) -> List[SearchResult]:
    """Web search over a provider ladder: Exa → Brave → DuckDuckGo.

    A rung is tried only when the one before it is unconfigured or came back
    empty, so a missing key or a provider outage degrades rather than failing.
    DuckDuckGo needs no key and always anchors the ladder, so behaviour with no
    keys configured is exactly what it was before Exa/Brave existed.

    Results are then junk-filtered and sorted by domain credibility before
    truncating to `limit` — the DDG rung over-fetches, so without this a buried
    authoritative result loses to whatever DDG ranked first. The sort is stable,
    so within a credibility tier the backend's own ranking holds.
    """
    if not query.strip():
        return []

    results: List[SearchResult] = []
    if EXA_API_KEY:
        results = await exa_search(query, limit, category=category)
        if not results:
            logger.info("Exa returned no usable results for %r; falling back.", query)
    if not results and BRAVE_API_KEY:
        results = await _brave_web_search(query, limit)
        if not results:
            logger.info("Brave returned no usable results for %r; falling back.", query)
    if not results:
        results = await _ddg_web_search(query, limit)

    from src.tools.search import _is_junk_domain
    from src.utils.source_scoring import evaluate_source
    filtered = [r for r in results if r.url and not _is_junk_domain(r.url)]
    filtered.sort(key=lambda r: evaluate_source(r.url, query)["score"], reverse=True)
    return filtered[:limit]
