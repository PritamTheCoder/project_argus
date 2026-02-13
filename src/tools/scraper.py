"""
Project Argus - Crawl4AI Web Scraper

Async scraper that fetches full page content, strips boilerplate,
and returns clean Markdown. Zero dependency on LLMs.
"""

import asyncio
import logging
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig
from src.config import SCRAPE_TIMEOUT

logger = logging.getLogger(__name__)


async def _scrape_single(
    crawler: AsyncWebCrawler,
    url: str,
    run_config: CrawlerRunConfig,
) -> dict:
    """
    Scrape a single URL and return a structured result dict.

    Returns:
        {
            "url": str,
            "content": str,      # Clean markdown text
            "success": bool,
            "error": str | None
        }
    """
    try:
        result = await asyncio.wait_for(
            crawler.arun(url=url, config=run_config),
            timeout=SCRAPE_TIMEOUT,
        )

        if result.success:
            content = result.markdown or ""

            logger.info(f"[OK] Scraped {url} ({len(content)} chars)")
            return {
                "url": url,
                "content": content.strip(),
                "success": True,
                "error": None,
            }
        else:
            error_msg = f"Crawl4AI returned failure for {url}"
            logger.warning(f"[X] {error_msg}")
            return {
                "url": url,
                "content": "",
                "success": False,
                "error": error_msg,
            }

    except asyncio.TimeoutError:
        error_msg = f"Timeout ({SCRAPE_TIMEOUT}s) scraping {url}"
        logger.warning(f"[X] {error_msg}")
        return {
            "url": url,
            "content": "",
            "success": False,
            "error": error_msg,
        }
    except Exception as e:
        error_msg = f"Error scraping {url}: {type(e).__name__}: {e}"
        logger.warning(f"[X] {error_msg}")
        return {
            "url": url,
            "content": "",
            "success": False,
            "error": error_msg,
        }


async def scrape_urls(urls: list[str]) -> list[dict]:
    """
    Scrape multiple URLs concurrently using Crawl4AI.

    Fetches the full DOM, strips boilerplate (navbars, ads, footers),
    and returns clean Markdown for each page. Failed URLs return
    structured errors instead of crashing the pipeline.

    Args:
        urls: List of URLs to scrape.

    Returns:
        List of result dicts, each with keys:
        - url (str): The original URL
        - content (str): Clean markdown text (empty on failure)
        - success (bool): Whether the scrape succeeded
        - error (str | None): Error message if failed, None if success
    """
    if not urls:
        return []

    logger.info(f"Scraping {len(urls)} URLs concurrently...")

    browser_config = BrowserConfig(
        headless=True,
        verbose=False,
    )

    run_config = CrawlerRunConfig(
        word_count_threshold=10,       # Skip pages with very little text
        excluded_tags=["nav", "footer", "header", "aside", "script", "style"],
        exclude_external_links=True,   # Keep output clean
    )

    async with AsyncWebCrawler(config=browser_config) as crawler:
        # Fire all scrapes concurrently with asyncio.gather
        tasks = [
            _scrape_single(crawler, url, run_config)
            for url in urls
        ]
        results = await asyncio.gather(*tasks, return_exceptions=False)

    successful = sum(1 for r in results if r["success"])
    logger.info(f"Scraping complete: {successful}/{len(urls)} succeeded")

    return results
