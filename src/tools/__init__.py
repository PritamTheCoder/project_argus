"""
Project Argus - Tools Package

Search, scrape, and refine. The refiner uses LangExtract (LLM-powered).
"""

from src.tools.search import search_ddg
from src.tools.scraper import scrape_urls
from src.tools.scout import run_scout
from src.tools.refiner import extract_facts

__all__ = ["search_ddg", "scrape_urls", "run_scout", "extract_facts"]
