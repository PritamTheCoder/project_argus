"""
Project Argus - Centralized Configuration

Loads environment variables from .env and exposes project-wide settings.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

# API Keys
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")
GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
NVIDIA_API_KEY: str = os.getenv("NVIDIA_API_KEY", "")

# Agent LLM Settings
LIBRARIAN_MODEL: str = os.getenv("LIBRARIAN_MODEL", "llama-3.3-70b-versatile")
LIBRARIAN_PROVIDER: str = os.getenv("LIBRARIAN_PROVIDER", "groq")

CRITIC_MODEL: str    = os.getenv("CRITIC_MODEL", "llama-3.3-70b-versatile")
CRITIC_PROVIDER: str = os.getenv("CRITIC_PROVIDER", "groq")

VERIFIER_MODEL: str  = os.getenv("VERIFIER_MODEL", "gemini-2.5-flash")
VERIFIER_PROVIDER: str = os.getenv("VERIFIER_PROVIDER", "gemini")

WRITER_MODEL: str    = os.getenv("WRITER_MODEL", "gemini-2.5-flash")
WRITER_PROVIDER: str = os.getenv("WRITER_PROVIDER", "gemini")

# Embedding Settings
EMBEDDING_MODE: str = os.getenv("EMBEDDING_MODE", "local") # "local" or "openai"
LOCAL_EMBEDDING_MODEL: str = os.getenv("LOCAL_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
MAX_CHUNK_TOKENS: int = int(os.getenv("MAX_CHUNK_TOKENS", "300"))
TOP_K_CHUNKS: int = int(os.getenv("TOP_K_CHUNKS", "15"))

# Refiner Settings
# Refiner sends entire scraped documents (~40K+ tokens). Groq free tier caps
# at 12K TPM, so Gemini (1M context) is the correct default here.
REFINER_MODEL: str = os.getenv("REFINER_MODEL", "gemini-2.5-flash")
REFINER_PROVIDER: str = os.getenv("REFINER_PROVIDER", "gemini")
REFINER_FALLBACK_MODEL: str = os.getenv("REFINER_FALLBACK_MODEL", "llama3-8b-8192")
REFINER_FALLBACK_PROVIDER: str = os.getenv("REFINER_FALLBACK_PROVIDER", "groq")

# Search Settings
MAX_SEARCH_RESULTS: int = int(os.getenv("MAX_SEARCH_RESULTS", "5"))

# ── Phase 2: Tool layer / multi-backend search ──────────────────────────────
# Brave Search API (optional). If unset, web_search falls back to DuckDuckGo.
BRAVE_API_KEY: str = os.getenv("BRAVE_API_KEY", "")
# Semantic Scholar API key (optional). Unset still works but is rate-limited.
SEMANTIC_SCHOLAR_API_KEY: str = os.getenv("SEMANTIC_SCHOLAR_API_KEY", "")
# Crossref "polite pool" contact (recommended by Crossref ToS).
CROSSREF_MAILTO: str = os.getenv("CROSSREF_MAILTO", "research@project-argus.local")
# Shared HTTP timeout (seconds) for tool/provider network calls.
TOOL_HTTP_TIMEOUT: int = int(os.getenv("TOOL_HTTP_TIMEOUT", "20"))
# Default result count for academic backends.
ACADEMIC_MAX_RESULTS: int = int(os.getenv("ACADEMIC_MAX_RESULTS", "8"))

# Acquisition agent (the tool-calling source gatherer). Must be a model that
# supports tool/function calling.
GATHERER_MODEL: str = os.getenv("GATHERER_MODEL", "gemini-2.5-flash")
GATHERER_PROVIDER: str = os.getenv("GATHERER_PROVIDER", "gemini")
# Max tool-calling rounds per sub-query before we stop (cost/loop safety).
GATHER_MAX_STEPS: int = int(os.getenv("GATHER_MAX_STEPS", "3"))

# Scraper Settings
SCRAPE_TIMEOUT: int = int(os.getenv("SCRAPE_TIMEOUT", "15"))

# Graph Orchestration
MAX_RESEARCH_LOOPS: int = int(os.getenv("MAX_RESEARCH_LOOPS", "2"))

# Junk Domain Blocklist
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

# Trusted Domain Allowlist
# Environments with extremely high signal-to-noise ratios.
TRUSTED_DOMAINS: set[str] = {
    "nature.com",
    "science.org",
    "ieee.org",
    "sciencedirect.com",
    "springer.com",
    "arxiv.org",
    "ncbi.nlm.nih.gov",
    "pubmed.ncbi.nlm.nih.gov",
    "iop.org", # Institute of Physics
    "acs.org", # American Chemical Society
    "rsc.org", # Royal Society of Chemistry
    "cell.com",
    "pnas.org", 
    "nejm.org",
    "bmj.com",
    "thelancet.com",
    "tandfonline.com",
    "sagepub.com",
    "wiley.com",
    "mit.edu",
    "stanford.edu",
    "harvard.edu",
    # Note: .gov and .edu top-level domains will be handled via regex/endswith in the filter logic
}

# Data Directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_SCRAPES_DIR = DATA_DIR / "raw_scrapes"
DB_DIR = DATA_DIR / "db"
DB_PATH = DB_DIR / "argus_checkpoints.db"
