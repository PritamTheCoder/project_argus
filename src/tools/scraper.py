import sqlite3
import os
"""
Project Argus - Crawl4AI Web Scraper

Async scraper that fetches full page content, strips boilerplate,
and returns clean Markdown. Zero dependency on LLMs.
"""

import asyncio
import logging
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig
from src.config import SCRAPE_TIMEOUT
from src.utils.cache import get_cached_markdown, set_cached_markdown, init_db
import re

logger = logging.getLogger(__name__)

# Ensure DB is initialized on import
init_db()

def _filter_long_content(content: str, query: str, max_length: int = 50000, intro_length: int = 10000) -> str:
    """
    If content exceeds max_length, aggressively filter it to reduce noise.
    Keeps the introduction (first `intro_length` chars).
    Then scans paragraphs for keywords from the query and builds up the remaining allowed budget.
    """
    if len(content) <= max_length:
        return content
        
    logger.info(f"    [!] Document length {len(content)} exceeds {max_length}. Applying pre-filter against query: '{query}'")
    
    # 1. Always keep the intro
    intro = content[:intro_length]
    remaining_budget = max_length - intro_length
    
    # 2. Extract keywords from query (ignoring common stop words roughly via length)
    keywords = [w.lower() for w in re.findall(r'\b\w+\b', query) if len(w) > 3]
    if not keywords:
        # If no good keywords, just truncate
        return intro + "\n\n... [CONTENT TRUNCATED FOR LENGTH] ..."
        
    # 3. Score paragraphs by keyword density
    rest_of_content = content[intro_length:]
    paragraphs = rest_of_content.split('\n\n')
    
    scored_paragraphs = []
    for i, p in enumerate(paragraphs):
        words = p.split()
        if not words:
            continue
            
        p_lower = p.lower()
        keyword_count = sum(p_lower.count(kw) for kw in keywords)
        
        if keyword_count > 0:
            density = keyword_count / len(words)
            scored_paragraphs.append((density, i, p))
            
    # Sort by density (highest first)
    scored_paragraphs.sort(key=lambda x: x[0], reverse=True)
    
    # Pick top paragraphs that fit in the budget
    selected_indices = []
    current_length = 0
    
    for density, original_idx, p in scored_paragraphs:
        # +2 for newlines
        if current_length + len(p) + 2 > remaining_budget:
            # Look for smaller paragraphs that might still fit
            continue
            
        selected_indices.append(original_idx)
        current_length += len(p) + 2
        
    # Sort selected indices back to original reading order for coherence
    selected_indices.sort()
    filtered_paragraphs = [paragraphs[i] for i in selected_indices]
            
    filtered_text = "\n\n".join(filtered_paragraphs)
    
    final_output = intro + f"\n\n... [SKIPPED {len(paragraphs) - len(filtered_paragraphs)} LESS RELEVANT PARAGRAPHS] ...\n\n" + filtered_text
    
    if len(selected_indices) < len(scored_paragraphs):
         final_output += "\n\n... [REMAINING LESS RELEVANT CONTENT TRUNCATED] ..."
         
    logger.info(f"    [!] Pre-filter complete. Final size: {len(final_output)} chars.")
    return final_output

async def _scrape_single(
    crawler: AsyncWebCrawler,
    url: str,
    run_config: CrawlerRunConfig,
    query: str = ""
) -> dict:
    """
    Scrape a single URL and return a structured result dict.
    Checks cache first to avoid reduntant network calls.
    Returns:
        {
            "url": str,
            "content": str,      # Clean markdown text
            "success": bool,
            "error": str | None
        }
    """
    # 1. Check persistent cache
    cached_content = get_cached_markdown(url)
    if cached_content:
        logger.info(f"[CACHE HIT] Loaded {url} from local DB ({len(cached_content)} chars)")
        
        # apply query pre-filter to cached content for the specific query
        if query and cached_content:
            cached_content = _filter_long_content(cached_content, query)
            
        return {
            "url": url,
            "content": cached_content.strip(),
            "success": True,
            "error": None,
        }

    # 2. Scrape live if not in cache
    try:
        result = await asyncio.wait_for(
            crawler.arun(url=url, config=run_config),
            timeout=SCRAPE_TIMEOUT,
        )

        if result.success:
            content = result.markdown or ""
            
            # Save raw content to cache before filtering
            if content:
                 set_cached_markdown(url, content)
            
            # Apply Wikipedia Noise Filter / Pre-filtering if query is provided
            if query and content:
                content = _filter_long_content(content, query)

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


async def scrape_urls(urls: list[str], query: str = "") -> list[dict]:
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
            _scrape_single(crawler, url, run_config, query)
            for url in urls
        ]
        results = await asyncio.gather(*tasks, return_exceptions=False)

    successful = sum(1 for r in results if r["success"])
    logger.info(f"Scraping complete: {successful}/{len(urls)} succeeded")

    return results
