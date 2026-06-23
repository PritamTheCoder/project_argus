"""Combines search + scrape into a single async pipeline: run_scout(query) -> clean JSON results."""

import logging
from src.tools.search import search_ddg
from src.tools.scraper import scrape_urls

logger = logging.getLogger(__name__)


async def run_scout(query: str, max_results: int | None = None) -> list[dict]:
    """
    Search DuckDuckGo for relevant URLs, scrape the full content of each page,
    and return a clean JSON array of results.

    Returns:
        A list of dicts, each containing:
        - url (str): Source URL
        - content (str): Clean markdown text from the page
        - success (bool): Whether the scrape succeeded
        - error (str | None): Error message if failed
    """
    logger.info(f" Scout pipeline started for: '{query}'")

    urls = await search_ddg(query, max_results=max_results)
    if not urls:
        logger.warning("No URLs found from search. Returning empty results.")
        return []

    logger.info(f" Found {len(urls)} URLs, starting scrape...")

    results = await scrape_urls(urls, query=query)

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
