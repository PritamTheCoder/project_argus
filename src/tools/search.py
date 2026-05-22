"""
Project Argus - DuckDuckGo Search Tool

Async wrapper around duckduckgo-search(ddgs) that returns clean URLs,
filtering out known junk domains.
"""

import logging
from urllib.parse import urlparse
from enum import Enum
from ddgs import DDGS
from src.config import MAX_SEARCH_RESULTS, JUNK_DOMAINS, TRUSTED_DOMAINS

logger = logging.getLogger(__name__)


class DomainPolicy(Enum):
    TRUSTED_ONLY = "trusted_only"
    TRUSTED_FIRST = "trusted_first"
    MIXED = "mixed"

def _is_junk_domain(url: str) -> bool:
    """Check if a URL belongs to a blocked junk domain."""
    try:
        hostname = urlparse(url).hostname or ""
        hostname = hostname.lower().removeprefix("www.")
        return hostname in JUNK_DOMAINS
    except Exception:
        return False

def _is_trusted_domain(url: str) -> bool:
    """Check if a URL belongs to a high-credibility allowlist."""
    try:
        hostname = urlparse(url).hostname or ""
        hostname = hostname.lower().removeprefix("www.")
        
        # Government and universities almost always contain high-signal facts
        if hostname.endswith(".gov") or hostname.endswith(".edu"):
            return True
            
        # Specific peer-reviewed / academic sources
        for trusted in TRUSTED_DOMAINS:
            if hostname == trusted or hostname.endswith("." + trusted):
                return True
                
        return False
    except Exception:
        return False


async def search_ddg(
    query: str,
    max_results: int | None = None,
    policy: DomainPolicy = DomainPolicy.MIXED
) -> list[str]:
    """
    Search DuckDuckGo and return a list of clean URLs.

    Args:
        query: The search query string.
        max_results: Maximum number of URLs to return (default from config).

    Returns:
        A list of URL strings, filtered to exclude junk domains.
    """
    if max_results is None:
        max_results = MAX_SEARCH_RESULTS

    logger.info(f"Searching DDG for: '{query}' (max {max_results} results)")

    try:
        # Fetch more results than needed to account for junk filtering
        fetch_count = max_results * 3

        with DDGS() as ddgs:
            raw_results = list(ddgs.text(query, max_results=fetch_count))

        # Extract URLs and apply domain policies
        trusted_urls: list[str] = []
        mixed_urls: list[str] = []
        
        for result in raw_results:
            url = result.get("href", "")
            if not url or _is_junk_domain(url):
                continue
                
            if _is_trusted_domain(url):
                trusted_urls.append(url)
            elif policy != DomainPolicy.TRUSTED_ONLY:
                mixed_urls.append(url)

        urls = []
        
        if policy == DomainPolicy.TRUSTED_ONLY:
            urls = trusted_urls[:max_results]
        elif policy == DomainPolicy.TRUSTED_FIRST:
            urls = trusted_urls
            # Fill remaining quota with clean mixed URLs
            if len(urls) < max_results:
                urls.extend(mixed_urls[:max_results - len(urls)])
            urls = urls[:max_results]
        else: # MIXED
            # Original behavior, strictly preserve DDG rank order
            urls = []
            for result in raw_results:
                 url = result.get("href", "")
                 if url and not _is_junk_domain(url):
                     if policy == DomainPolicy.MIXED:
                         urls.append(url)
                 if len(urls) >= max_results:
                     break

        logger.info(f"DDG ({policy.value}) returned {len(urls)} URLs")
        return urls

    except Exception as e:
        logger.error(f"DuckDuckGo search failed for '{query}': {e}")
        return []
