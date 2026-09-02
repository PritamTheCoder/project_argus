import sqlite3
import os
"""Async scraper that fetches full page content, strips boilerplate, and returns clean Markdown."""

import asyncio
import logging
import warnings
from contextlib import asynccontextmanager
import aiohttp
import fitz  # PyMuPDF
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig
from src.config import SCRAPE_TIMEOUT, CRAWLER_CLOSE_TIMEOUT
from src.utils.cache import get_cached_markdown, set_cached_markdown, init_db
import re

try:
    from playwright._impl._errors import Error as PlaywrightError
except ImportError:
    PlaywrightError = OSError  # safe fallback

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

    intro = content[:intro_length]
    remaining_budget = max_length - intro_length

    # Filter words >3 chars as a cheap stop-word heuristic (avoids a stop-word list).
    keywords = [w.lower() for w in re.findall(r'\b\w+\b', query) if len(w) > 3]
    if not keywords:
        return intro + "\n\n... [CONTENT TRUNCATED FOR LENGTH] ..."

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

    scored_paragraphs.sort(key=lambda x: x[0], reverse=True)

    selected_indices = []
    current_length = 0

    for density, original_idx, p in scored_paragraphs:
        if current_length + len(p) + 2 > remaining_budget:  # +2 for the joining newlines
            continue

        selected_indices.append(original_idx)
        current_length += len(p) + 2

    # Restore original reading order so the result stays coherent.
    selected_indices.sort()
    filtered_paragraphs = [paragraphs[i] for i in selected_indices]
            
    filtered_text = "\n\n".join(filtered_paragraphs)
    
    final_output = intro + f"\n\n... [SKIPPED {len(paragraphs) - len(filtered_paragraphs)} LESS RELEVANT PARAGRAPHS] ...\n\n" + filtered_text
    
    if len(selected_indices) < len(scored_paragraphs):
         final_output += "\n\n... [REMAINING LESS RELEVANT CONTENT TRUNCATED] ..."
         
    logger.info(f"    [!] Pre-filter complete. Final size: {len(final_output)} chars.")
    return final_output

async def _download_and_parse_pdf_in_memory(url: str, session: aiohttp.ClientSession) -> dict:
    """
    Downloads a PDF directly into memory (no lingering disk files)
    and extracts all text using PyMuPDF (fitz).
    """
    try:
        logger.info(f"    [PDF Route] Downloading PDF into memory from {url}...")
        async with session.get(url, timeout=SCRAPE_TIMEOUT) as response:
            if response.status != 200:
                error_msg = f"Failed to download PDF, status code {response.status}"
                logger.warning(f"[X] {error_msg}")
                return {"url": url, "content": "", "success": False, "error": error_msg}

            pdf_bytes = await response.read()

        logger.info(f"    [PDF Route] Parsing {len(pdf_bytes)} bytes of PDF in memory...")
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")

        pdf_text = []
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text = page.get_text("text")
            pdf_text.append(text)

        full_text = "\n\n".join(pdf_text)

        doc.close()
        del pdf_bytes
        
        if not full_text.strip():
            error_msg = "PDF downloaded but extracted text is empty (might be scanned images)."
            logger.warning(f"[X] {error_msg}")
            return {"url": url, "content": "", "success": False, "error": error_msg}
            
        logger.info(f"[OK] Parsed PDF {url} ({len(full_text)} chars extracted)")
        return {
            "url": url,
            "content": full_text.strip(),
            "success": True,
            "error": None,
        }
            
    except asyncio.TimeoutError:
        error_msg = f"Timeout ({SCRAPE_TIMEOUT}s) downloading PDF from {url}"
        logger.warning(f"[X] {error_msg}")
        return {"url": url, "content": "", "success": False, "error": error_msg}
    except fitz.FileDataError as e:
        error_msg = f"Corrupted or invalid PDF from {url}: {e}"
        logger.warning(f"[X] {error_msg}")
        return {"url": url, "content": "", "success": False, "error": error_msg}
    except Exception as e:
        error_msg = f"Error extracting PDF from {url}: {type(e).__name__}: {e}"
        logger.warning(f"[X] {error_msg}")
        return {"url": url, "content": "", "success": False, "error": error_msg}

async def _scrape_single(
    crawler: AsyncWebCrawler,
    url: str,
    run_config: CrawlerRunConfig,
    query: str = ""
) -> dict:
    """
    Scrape a single URL and return a structured result dict.
    Checks cache first to avoid redundant network calls.

    Uses Playwright-native timeouts (via CrawlerRunConfig.page_timeout)
    instead of asyncio.wait_for to prevent orphaned Futures that cause
    'Future exception was never retrieved' warnings.

    Returns:
        {
            "url": str,
            "content": str,      # Clean markdown text
            "success": bool,
            "error": str | None
        }
    """
    cached_content = get_cached_markdown(url)
    if cached_content:
        logger.info(f"[CACHE HIT] Loaded {url} from local DB ({len(cached_content)} chars)")

        if query and cached_content:
            cached_content = _filter_long_content(cached_content, query)

        return {
            "url": url,
            "content": cached_content.strip(),
            "success": True,
            "error": None,
        }

    try:
        # Pre-flight via aiohttp to route PDFs vs. HTML before invoking Playwright.
        async with aiohttp.ClientSession() as session:
            try:
                async with session.head(url, allow_redirects=True, timeout=10) as head_resp:
                    content_type = head_resp.headers.get("Content-Type", "").lower()

                # Some servers don't respond properly to HEAD; fall back to GET.
                if not content_type or head_resp.status != 200:
                    async with session.get(url, allow_redirects=True, timeout=10) as get_resp:
                        content_type = get_resp.headers.get("Content-Type", "").lower()
            except Exception as e:
                logger.warning(f"    [!] Pre-flight check failed for {url} ({e}). Defaulting to HTML crawler.")
                content_type = "text/html"

        if "application/pdf" in content_type or url.lower().endswith(".pdf"):
            logger.info(f"    [!] Detected PDF artifact at {url}. Bypassing Playwright crawler.")
            async with aiohttp.ClientSession() as session:
                pdf_result = await _download_and_parse_pdf_in_memory(url, session)
                
                if pdf_result["success"] and pdf_result["content"]:
                    content = pdf_result["content"]
                    set_cached_markdown(url, content)
                    
                    if query:
                        content = _filter_long_content(content, query)
                        
                    return {
                        "url": url,
                        "content": content.strip(),
                        "success": True,
                        "error": None,
                    }
                else:
                    return pdf_result

        # Let Playwright handle the timeout natively — avoids orphaned
        # Futures that asyncio.wait_for would create when it cancels
        # the coroutine while Playwright navigation is still in-flight.
        result = await crawler.arun(url=url, config=run_config)

        if result.success:
            content = result.markdown or ""

            # Some pages render near-empty on first load; retry headed.
            if len(content) < 1000:
                logger.info(f"[!] Scrape of {url} yielded only {len(content)} chars. Retrying with headless=False...")
                try:
                    retry_browser_config = BrowserConfig(headless=False, verbose=False)
                    retry_run_config = CrawlerRunConfig(
                        word_count_threshold=run_config.word_count_threshold,
                        excluded_tags=run_config.excluded_tags,
                        exclude_external_links=run_config.exclude_external_links,
                        page_timeout=SCRAPE_TIMEOUT * 1000,  # ms
                    )
                    async with AsyncWebCrawler(config=retry_browser_config) as retry_crawler:
                        retry_result = await retry_crawler.arun(
                            url=url, config=retry_run_config
                        )
                        # Settle pending network requests/frames
                        await asyncio.sleep(1.0)
                    if retry_result.success and retry_result.markdown and len(retry_result.markdown) > len(content):
                        content = retry_result.markdown
                        logger.info(f"    [+] Retry successful! Extracted {len(content)} chars.")
                    else:
                        logger.warning(f"    [-] Retry failed or yielded shorter content. Keeping original.")
                except (PlaywrightError, asyncio.TimeoutError) as e:
                    logger.warning(f"    [-] Retry navigation error: {e}")
                except Exception as e:
                    logger.warning(f"    [-] Retry failed with error: {e}")

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

    except PlaywrightError as e:
        # Playwright navigation errors (ERR_ABORTED, frame detached, etc.)
        # Caught explicitly to prevent them from becoming orphaned Futures.
        error_msg = f"Navigation error for {url}: {e}"
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


def _browser_config() -> BrowserConfig:
    return BrowserConfig(headless=True, verbose=False)


@asynccontextmanager
async def shared_crawler():
    """One browser instance for a whole batch of scrape_urls() calls, instead
    of one Chromium process per call — avoids exhausting OS process limits
    when Scout scrapes several sub-queries concurrently.

    Uses explicit start()/close() rather than `async with AsyncWebCrawler(...)`
    so close() can be time-boxed: Playwright's browser-close handshake can hang
    indefinitely without raising. On timeout we log and move on — a leaked
    browser process is a smaller problem than a permanently hung job.
    """
    crawler = AsyncWebCrawler(config=_browser_config())
    await crawler.start()
    try:
        yield crawler
    finally:
        try:
            await asyncio.wait_for(crawler.close(), timeout=CRAWLER_CLOSE_TIMEOUT)
        except asyncio.TimeoutError:
            logger.warning(f"Browser close did not finish within {CRAWLER_CLOSE_TIMEOUT}s — continuing without it.")
        except Exception as e:
            logger.warning(f"Browser close raised {type(e).__name__}: {e}")


async def scrape_urls(urls: list[str], query: str = "", crawler: AsyncWebCrawler | None = None) -> list[dict]:
    """
    Scrape multiple URLs concurrently using Crawl4AI.

    Fetches the full DOM, strips boilerplate (navbars, ads, footers),
    and returns clean Markdown for each page. Failed URLs return
    structured errors instead of crashing the pipeline.

    ``crawler``: reuse an existing ``AsyncWebCrawler`` instead of opening a new
    one. Omit for a one-off scrape; concurrent callers should share one via
    ``shared_crawler()``.

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

    run_config = CrawlerRunConfig(
        word_count_threshold=10,
        excluded_tags=["nav", "footer", "header", "aside", "script", "style"],
        exclude_external_links=True,
        page_timeout=SCRAPE_TIMEOUT * 1000,  # ms — Playwright-native timeout
    )

    if crawler is not None:
        raw_results = await _scrape_batch(crawler, urls, run_config, query)
    else:
        async with shared_crawler() as owned_crawler:
            raw_results = await _scrape_batch(owned_crawler, urls, run_config, query)

    results = []
    for url, res in zip(urls, raw_results):
        if isinstance(res, Exception):
            logger.warning(f"[X] Unhandled exception during scrape of {url}: {type(res).__name__}: {res}")
            results.append({
                "url": url,
                "content": "",
                "success": False,
                "error": f"{type(res).__name__}: {res}"
            })
        else:
            results.append(res)

    successful = sum(1 for r in results if r.get("success", False))
    logger.info(f"Scraping complete: {successful}/{len(urls)} succeeded")

    return results


async def _scrape_batch(crawler: AsyncWebCrawler, urls: list[str], run_config: CrawlerRunConfig, query: str) -> list:
    tasks = [_scrape_single(crawler, url, run_config, query) for url in urls]
    # Use return_exceptions=True to prevent a single TargetClosedError from crashing the batch
    raw_results = await asyncio.gather(*tasks, return_exceptions=True)
    # Give Playwright time to clean up frames / pending navigations
    await asyncio.sleep(1.0)
    return raw_results
