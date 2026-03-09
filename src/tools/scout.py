"""
Project Argus - Scout Pipeline

Combines search + scrape into a single async pipeline.
This is the Phase 1 deliverable: run_scout(query) → clean JSON results.
"""

import logging
from src.tools.search import search_ddg
from src.tools.scraper import scrape_urls

logger = logging.getLogger(__name__)


async def run_scout(query: str, max_results: int | None = None) -> list[dict]:
    """
    Execute the full Scout pipeline: Search → Scrape → Return.

    Given a research query,
    it searches DuckDuckGo for relevant URLs, scrapes the full content
    of each page, and returns a clean JSON array of results.

    Args:
        query: The research query string (e.g., "solid state batteries 2026").
        max_results: Maximum number of sources to retrieve (default from config).

    Returns:
        A list of dicts, each containing:
        - url (str): Source URL
        - content (str): Clean markdown text from the page
        - success (bool): Whether the scrape succeeded
        - error (str | None): Error message if failed

    Example:
        >>> import asyncio
        >>> results = asyncio.run(run_scout("solid state batteries 2026"))
        >>> print(f"Got {len(results)} results")
        >>> for r in results:
        ...     print(f"  {r['url']}: {len(r['content'])} chars")
    """
    logger.info(f" Scout pipeline started for: '{query}'")

    # Search for relevant URLs
    urls = await search_ddg(query, max_results=max_results)
    if not urls:
        logger.warning("No URLs found from search. Returning empty results.")
        return []

    logger.info(f" Found {len(urls)} URLs, starting scrape...")

    # Scrape all URLs concurrently, passing query for pre-filtering
    results = await scrape_urls(urls, query=query)

    # Log summary
    successful = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]

    logger.info(
        f" Scout pipeline complete: "
        f"{len(successful)} succeeded, {len(failed)} failed"
    )

    if failed:
        for r in failed:
            logger.debug(f"  Failed: {r['url']} — {r['error']}")

    return results
