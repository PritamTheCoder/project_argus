"""
Project Argus - Centralized Configuration

Loads environment variables from .env and exposes project-wide settings.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# ── Load .env from project root ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

# ── API Keys ─────────────────────────────────────────────────────────────────
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")
GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")

# ── Agent LLM Settings ──────────────────────────────────────────────────────
LIBRARIAN_MODEL: str = os.getenv("LIBRARIAN_MODEL", "gemini-2.5-flash")
CRITIC_MODEL: str    = os.getenv("CRITIC_MODEL", "gemini-2.5-flash")
WRITER_MODEL: str    = os.getenv("WRITER_MODEL", "gemini-2.5-flash")

# ── Embedding Settings ───────────────────────────────────────────────────────
EMBEDDING_MODE: str = os.getenv("EMBEDDING_MODE", "local") # "local" or "openai"
LOCAL_EMBEDDING_MODEL: str = os.getenv("LOCAL_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
MAX_CHUNK_TOKENS: int = int(os.getenv("MAX_CHUNK_TOKENS", "300"))
TOP_K_CHUNKS: int = int(os.getenv("TOP_K_CHUNKS", "8"))

# ── Refiner Settings ────────────────────────────────────────────────────────
REFINER_MODEL: str = os.getenv("REFINER_MODEL", "gemini-2.5-flash")
REFINER_FALLBACK_MODEL: str = os.getenv("REFINER_FALLBACK_MODEL", "llama3-8b-8192")
REFINER_PROVIDERS: list[str] = ["gemini", "groq"]

# ── Search Settings ──────────────────────────────────────────────────────────
MAX_SEARCH_RESULTS: int = int(os.getenv("MAX_SEARCH_RESULTS", "5"))

# ── Scraper Settings ────────────────────────────────────────────────────────
SCRAPE_TIMEOUT: int = int(os.getenv("SCRAPE_TIMEOUT", "15"))

# ── Graph Orchestration ─────────────────────────────────────────────────────
MAX_RESEARCH_LOOPS: int = int(os.getenv("MAX_RESEARCH_LOOPS", "2"))

# ── Junk Domain Blocklist ────────────────────────────────────────────────────
# These domains return low-quality, opinion-heavy, or paywalled content
# that degrades research output.
JUNK_DOMAINS: set[str] = {
    "pinterest.com",
    "pinterest.co.uk",
    "quora.com",
    "reddit.com",
    "facebook.com",
    "instagram.com",
    "twitter.com",
    "x.com",
    "tiktok.com",
    "youtube.com",
    "linkedin.com",
    "medium.com",        # often paywalled
    "researchgate.net",  # often paywalled
}

# ── Data Directories ────────────────────────────────────────────────────────
DATA_DIR = PROJECT_ROOT / "data"
RAW_SCRAPES_DIR = DATA_DIR / "raw_scrapes"
DB_DIR = DATA_DIR / "db"
DB_PATH = DB_DIR / "argus_checkpoints.db"
