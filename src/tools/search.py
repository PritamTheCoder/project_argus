"""
Project Argus - DuckDuckGo Search Tool

Async wrapper around duckduckgo-search(ddgs) that returns clean URLs,
filtering out known junk domains.
"""

import logging
from urllib.parse import urlparse
from ddgs import DDGS
from src.config import MAX_SEARCH_RESULTS, JUNK_DOMAINS

logger = logging.getLogger(__name__)


def _is_junk_domain(url: str) -> bool:
    """Check if a URL belongs to a blocked junk domain."""
    try:
        hostname = urlparse(url).hostname or ""
        # Strip 'www.' prefix for matching
        hostname = hostname.lower().removeprefix("www.")
        return hostname in JUNK_DOMAINS
    except Exception:
        return False


async def search_ddg(
    query: str,
    max_results: int | None = None,
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

        # Extract URLs and filter junk domains
        urls: list[str] = []
        for result in raw_results:
            url = result.get("href", "")
            if url and not _is_junk_domain(url):
                urls.append(url)
            if len(urls) >= max_results:
                break

        logger.info(f"DDG returned {len(urls)} clean URLs (filtered from {len(raw_results)} raw)")
        return urls

    except Exception as e:
        logger.error(f"DuckDuckGo search failed for '{query}': {e}")
        return []
