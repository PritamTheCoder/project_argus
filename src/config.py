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
# NVIDIA NIM (OpenAI-compatible) endpoint. Each model below has its own key
# (own free-tier quota) but shares this base URL.
NVIDIA_BASE_URL: str = os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")

# Nemotron-3 Ultra (reasoning). Writer's primary when the key is set.
NVIDIA_API_NEMOTRON3_KEY: str = os.getenv("NVIDIA_API_NEMOTRON3_KEY", "")
NEMOTRON_MODEL: str = os.getenv("NEMOTRON_MODEL", "nvidia/nemotron-3-ultra-550b-a55b")
# Unused: NVIDIA's model runner currently rejects this param. See llm_factory._nim_spec.
NEMOTRON_REASONING_BUDGET: int = int(os.getenv("NEMOTRON_REASONING_BUDGET", "4096"))

# StepFun Step-3.7 Flash (multimodal). Dead as of 2026-08-28 (410, end of life).
STEP_API_KEY: str = os.getenv("STEP_3.7_API_KEY", "")
STEP_MODEL: str = os.getenv("STEP_MODEL", "stepfun-ai/step-3.7-flash")

# Kimi K2.6 via NVIDIA NIM. Not entitled on this account (404).
KIMI_API_KEY: str = os.getenv("KIMI_NVIDIA_KEY", "")
KIMI_MODEL: str = os.getenv("KIMI_MODEL", "moonshotai/kimi-k2.6")

# Z.AI GLM-5.2 via NVIDIA NIM. Dead as of 2026-08-21 (410, end of life).
GLM_API_KEY: str = os.getenv("ZLM_NVIDIA_KEY", "")
GLM_MODEL: str = os.getenv("GLM_MODEL", "z-ai/glm-5.2")

# ── Default Gemini model ─────────────────────────────────────────────────────
GEMINI_DEFAULT_MODEL: str = os.getenv("GEMINI_DEFAULT_MODEL", "gemini-3.1-flash-lite")

# Agent LLM Settings.
# Shared fallback for the Groq-primary structured nodes. Gemini is the only
# working alternate right now (Step/GLM/Kimi are all dead or not entitled —
# see PRODUCTION_GRADE_AND_TOOL_USE_PLAN.md for the full provider audit).
_STRUCTURED_FALLBACK = f"{GEMINI_DEFAULT_MODEL}:gemini"

LIBRARIAN_MODEL: str = os.getenv("LIBRARIAN_MODEL", "openai/gpt-oss-120b")
LIBRARIAN_PROVIDER: str = os.getenv("LIBRARIAN_PROVIDER", "groq")
LIBRARIAN_FALLBACK_CHAIN: str = os.getenv("LIBRARIAN_FALLBACK_CHAIN", _STRUCTURED_FALLBACK)

CRITIC_MODEL: str    = os.getenv("CRITIC_MODEL", "openai/gpt-oss-120b")
CRITIC_PROVIDER: str = os.getenv("CRITIC_PROVIDER", "groq")
CRITIC_FALLBACK_CHAIN: str = os.getenv("CRITIC_FALLBACK_CHAIN", _STRUCTURED_FALLBACK)

# Verifier: high-frequency, small batches (<=30 facts). Gemini's free tier is
# too small to be primary, so it's the fallback only.
VERIFIER_MODEL: str  = os.getenv("VERIFIER_MODEL", "openai/gpt-oss-120b")
VERIFIER_PROVIDER: str = os.getenv("VERIFIER_PROVIDER", "groq")
VERIFIER_FALLBACK_CHAIN: str = os.getenv("VERIFIER_FALLBACK_CHAIN", _STRUCTURED_FALLBACK)

# Writer: long-form synthesis from evidence the Verifier has already checked.
# It assembles prose from pre-verified bullets rather than reasoning its way to
# new conclusions, so a fast model is the right tool; a large reasoning model
# costs minutes per report for no quality gain.
WRITER_MODEL: str    = os.getenv("WRITER_MODEL", "openai/gpt-oss-120b")
WRITER_PROVIDER: str = os.getenv("WRITER_PROVIDER", "groq")
# Nemotron sits on the ladder rather than at the front: available when Groq is
# rate-limited, without paying its latency on every run.
_writer_fallback = (
    f"{NEMOTRON_MODEL}:nemotron,{_STRUCTURED_FALLBACK}"
    if NVIDIA_API_NEMOTRON3_KEY else _STRUCTURED_FALLBACK
)
WRITER_FALLBACK_CHAIN: str = os.getenv("WRITER_FALLBACK_CHAIN", _writer_fallback)

# Embedding Settings
EMBEDDING_MODE: str = os.getenv("EMBEDDING_MODE", "local") # "local" or "openai"
LOCAL_EMBEDDING_MODEL: str = os.getenv("LOCAL_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
MAX_CHUNK_TOKENS: int = int(os.getenv("MAX_CHUNK_TOKENS", "300"))
TOP_K_CHUNKS: int = int(os.getenv("TOP_K_CHUNKS", "15"))

# Refiner Settings
# Sends whole scraped documents, so the model must be large-context. Groq
# gpt-oss-120b (128K) is primary; Gemini's free tier can't sustain this as primary.
REFINER_MODEL: str = os.getenv("REFINER_MODEL", "openai/gpt-oss-120b")
REFINER_PROVIDER: str = os.getenv("REFINER_PROVIDER", "groq")
REFINER_FALLBACK_CHAIN: str = os.getenv("REFINER_FALLBACK_CHAIN", f"{GEMINI_DEFAULT_MODEL}:gemini")

# Extraction is split into batches sized to fit the primary's per-minute TOKEN
# budget. Packing every scraped document into one prompt fits the model's
# context but breaches free-tier TPM limits, and the whole payload is then
# re-sent to the fallback rung. Batching bounds that: a failed batch costs one
# batch, not the run. ~4 chars/token, so 32K chars ≈ 8K tokens per request.
REFINER_BATCH_MAX_CHARS: int = int(os.getenv("REFINER_BATCH_MAX_CHARS", "32000"))
# Ceiling on a single document's contribution, so one huge page can't by itself
# produce a batch that breaches the budget.
REFINER_MAX_DOC_CHARS: int = int(os.getenv("REFINER_MAX_DOC_CHARS", "32000"))

# ── Client-side rate limiting ────────────────────────────────────────────────
# Proactive per-provider throttle (requests/minute), so we stay under free-tier
# limits instead of reactively eating 429s.
# GEMINI_MAX_RETRIES=1 means no in-SDK retries: on a 429/503 we want to fail
# fast to the next fallback provider, not have the SDK sleep and block the run.
GEMINI_MAX_RETRIES: int = int(os.getenv("GEMINI_MAX_RETRIES", "1"))
GEMINI_TIMEOUT: int = int(os.getenv("GEMINI_TIMEOUT", "60"))  # hard per-request ceiling
GEMINI_RPM: int = int(os.getenv("GEMINI_RPM", "25"))
GROQ_RPM: int = int(os.getenv("GROQ_RPM", "30"))  # per key; multiplied by the key pool
OPENAI_RPM: int = int(os.getenv("OPENAI_RPM", "60"))
NVIDIA_RPM: int = int(os.getenv("NVIDIA_RPM", "30"))
NEMOTRON_RPM: int = int(os.getenv("NEMOTRON_RPM", "8"))  # large reasoning model, keep conservative
STEP_RPM: int = int(os.getenv("STEP_RPM", "10"))  # unused, model is dead
KIMI_RPM: int = int(os.getenv("KIMI_RPM", "10"))  # unused, not entitled
GLM_RPM: int = int(os.getenv("GLM_RPM", "20"))  # unused, model is dead
# How many calls fire immediately before the limiter throttles to steady RPM.
RATE_LIMIT_BURST: int = int(os.getenv("RATE_LIMIT_BURST", "8"))
# Client-side timeout for NIM calls, so a stuck request fails over quickly
# instead of waiting on NVIDIA's own ~5 min gateway timeout.
NIM_TIMEOUT: int = int(os.getenv("NIM_TIMEOUT", "120"))

# Search Settings
MAX_SEARCH_RESULTS: int = int(os.getenv("MAX_SEARCH_RESULTS", "5"))

# ── Tool layer / multi-backend search ───────────────────────────────────────
# web_search tries these in order: Exa → Brave → DuckDuckGo. Each rung is used
# only if the one before it is unconfigured or returns nothing, so missing keys
# degrade instead of failing. DuckDuckGo needs no key and is the final backstop.
#
# Exa (neural search over a curated index). Its `category` filter is the reason
# it leads the ladder: asking for a "financial report" is a query parameter
# rather than a hope about ranking. Free tier is 20k requests/month.
EXA_API_KEY: str = os.getenv("EXA_API_KEY", "")
# How much page text Exa returns inline. Text it supplies skips the scraper
# entirely, so this trades tokens against scrape latency.
EXA_TEXT_CHARS: int = int(os.getenv("EXA_TEXT_CHARS", "4000"))
# "text" (default) or "highlights". Text gives continuous prose, which the
# verbatim-quote grounding in the Verifier needs; highlights are cheaper per
# result but fragmentary. See _exa_contents_payload().
EXA_CONTENT_MODE: str = os.getenv("EXA_CONTENT_MODE", "text").strip().lower()
# auto | fast | instant | deep-lite | deep | deep-reasoning. "auto" balances
# relevance and latency (~1s). instant/fast/auto/deep-lite all bill at the same
# rate, so `fast` is free speed — but the `deep` variants cost ~2x per request.
EXA_SEARCH_TYPE: str = os.getenv("EXA_SEARCH_TYPE", "auto").strip().lower()
# Results per Exa request. Exa bills $7/1k for the search plus $1/1k *per page*
# for contents, so each extra result we don't actually use is pure waste. The
# Scout only registers a handful of sources per sub-query, so 5 is plenty.
EXA_MAX_RESULTS: int = int(os.getenv("EXA_MAX_RESULTS", "5"))
# Hours an Exa response stays cached. Re-running a research query is the normal
# development loop, and each repeat is billed — caching makes it free. 0 = off.
EXA_CACHE_TTL_HOURS: int = int(os.getenv("EXA_CACHE_TTL_HOURS", "24"))

# Brave Search API (optional). If unset, web_search falls back to DuckDuckGo.
BRAVE_API_KEY: str = os.getenv("BRAVE_API_KEY", "")

# SEC EDGAR: free, no key, but SEC policy requires a descriptive User-Agent with
# a contact address, and rate-limits to 10 req/s.
EDGAR_USER_AGENT: str = os.getenv("EDGAR_USER_AGENT", "ProjectArgus research@project-argus.local")
EDGAR_MAX_RESULTS: int = int(os.getenv("EDGAR_MAX_RESULTS", "10"))
# Concurrent primary_doc.xml fetches when enriching filings. SEC allows
# 10 req/s; stay well under it since scrapes may run alongside.
EDGAR_ENRICH_CONCURRENCY: int = int(os.getenv("EDGAR_ENRICH_CONCURRENCY", "4"))
# Semantic Scholar API key (optional). Unset still works but is rate-limited.
SEMANTIC_SCHOLAR_API_KEY: str = os.getenv("SEMANTIC_SCHOLAR_API_KEY", "")
# Crossref "polite pool" contact (recommended by Crossref ToS).
CROSSREF_MAILTO: str = os.getenv("CROSSREF_MAILTO", "research@project-argus.local")
# Shared HTTP timeout (seconds) for tool/provider network calls.
TOOL_HTTP_TIMEOUT: int = int(os.getenv("TOOL_HTTP_TIMEOUT", "20"))
# Default result count for academic backends.
ACADEMIC_MAX_RESULTS: int = int(os.getenv("ACADEMIC_MAX_RESULTS", "8"))

# Tool search: when the tool catalog exceeds this many tools, the acquisition
# agent binds only the top-N most relevant tools rather than every tool. Below
# this, all tools are bound (no embedding work).
TOOL_SEARCH_MAX_TOOLS: int = int(os.getenv("TOOL_SEARCH_MAX_TOOLS", "6"))

# MCP (Model Context Protocol) servers to load tools from at startup. JSON map
# of {server_name: {transport, command/args or url}}. Empty = MCP disabled.
# Example: {"fs": {"transport": "stdio", "command": "npx",
#                  "args": ["-y", "@modelcontextprotocol/server-filesystem", "/data"]}}
MCP_SERVERS_JSON: str = os.getenv("MCP_SERVERS_JSON", "")

# Acquisition agent (the tool-calling source gatherer). Must support tool calling.
GATHERER_MODEL: str = os.getenv("GATHERER_MODEL", "openai/gpt-oss-120b")
GATHERER_PROVIDER: str = os.getenv("GATHERER_PROVIDER", "groq")
GATHERER_FALLBACK_CHAIN: str = os.getenv("GATHERER_FALLBACK_CHAIN", _STRUCTURED_FALLBACK)
# Max tool-calling rounds per sub-query before stopping (cost/loop safety).
GATHER_MAX_STEPS: int = int(os.getenv("GATHER_MAX_STEPS", "2"))
# How many sub-queries the Scout gathers+scrapes concurrently. Overlaps the
# network-bound work (tool calls + scraping) across the plan's queries; the
# per-provider rate limiter still caps actual API RPM. Keep modest so we don't
# open too many simultaneous scrapes.
SCOUT_CONCURRENCY: int = int(os.getenv("SCOUT_CONCURRENCY", "4"))
# kg_lookup scope. Default "session": the memory tool sees only THIS run's facts,
# consistent with the session-scoped Critic/Reflector retrieval (no cross-run
# contamination). Set "global" to opt into cross-run memory (the future
# memory-first path — reuses prior runs' verified facts).
KG_LOOKUP_GLOBAL: bool = os.getenv("KG_LOOKUP_SCOPE", "session").lower() == "global"

# Scraper Settings
SCRAPE_TIMEOUT: int = int(os.getenv("SCRAPE_TIMEOUT", "15"))
# Playwright's browser-close handshake can hang indefinitely with no error
# (a crawl4ai/Playwright issue, more likely under concurrent scraping). This
# bounds how long we wait for it before giving up and moving on, so a stuck
# shutdown can't stall an entire research job.
CRAWLER_CLOSE_TIMEOUT: int = int(os.getenv("CRAWLER_CLOSE_TIMEOUT", "15"))

# Verifier Settings
# Facts per verification batch. Larger = fewer LLM calls (less RPM pressure) but
# risks "lost in the middle" degradation on very long batches. 30 is the sweet spot
# for flash-class models with a 128K+ context window.
VERIFY_BATCH_SIZE: int = int(os.getenv("VERIFY_BATCH_SIZE", "30"))
# Hard cap on facts sent to the verifier per run. Pre-dedup facts above this
# threshold are dropped (lowest-credibility sources dropped first). Keeps call
# count bounded regardless of how many pages the scout scraped.
MAX_FACTS_TO_VERIFY: int = int(os.getenv("MAX_FACTS_TO_VERIFY", "150"))

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
    "harvard.edu" # Note: .gov and .edu top-level domains will be handled via regex/endswith in the filter logic
}

# Major News Outlets — high-credibility non-academic journalism, staff-reported.
MAJOR_NEWS_DOMAINS: set[str] = {
    "reuters.com", "bloomberg.com", "bbc.com", "bbc.co.uk",
    "nytimes.com", "washingtonpost.com", "wsj.com",
    "cnbc.com", "nbcnews.com", "apnews.com",
    "theguardian.com", "ft.com", "economist.com",
    "sciencedaily.com", "arstechnica.com",
    # Business/financial desks — the tier to trust for market reporting.
    "fortune.com", "theglobeandmail.com", "theinformation.com",
    "barrons.com", "axios.com", "politico.com", "latimes.com",
    "cnn.com", "abcnews.go.com", "cbsnews.com", "time.com",
}

# Reputable Industry / Market Research firms.
INDUSTRY_DOMAINS: set[str] = {
    "marketsandmarkets.com", "grandviewresearch.com",
    "researchandmarkets.com", "mordorintelligence.com",
    "statista.com", "iea.org", "irena.org",
    "mckinsey.com", "bcg.com", "deloitte.com",
    "pwc.com", "kpmg.com",
    "yahoo.com",  # yahoo finance/news
    "cars.com", "notebookcheck.net",
    # Company / funding databases. Professionally maintained and routinely cited
    # by financial press — aggregators rather than primary sources, so they sit
    # at the industry tier, not the government/academic one.
    "crunchbase.com", "cbinsights.com", "pitchbook.com",
    "tracxn.com", "dealroom.co", "sacra.com", "caplight.com",
    # Trade and business press. Real reporting, but either topic-specialist or
    # carrying contributor content of uneven quality (Forbes' /sites/ network,
    # Business Insider), so they sit a tier below staff-reported major news.
    "forbes.com", "techcrunch.com", "businessinsider.com",
    "morningstar.com", "marketwatch.com", "spacenews.com",
    "theverge.com", "wired.com", "engadget.com", "zdnet.com",
    "investopedia.com", "nasdaq.com",
}

# A source's registrable domain can host at most this many sources per run, so
# one site (or a network of subpages on it) can't crowd out source diversity.
MAX_SOURCES_PER_DOMAIN: int = int(os.getenv("MAX_SOURCES_PER_DOMAIN", "2"))
# Domains scoring at or above this are exempt from that cap — sec.gov and
# journal publishers legitimately supply many documents, and capping them
# throws away primary sources to make room for blogs.
AUTHORITATIVE_CREDIBILITY: float = float(os.getenv("AUTHORITATIVE_CREDIBILITY", "0.8"))

# Below this average source credibility, the research loop re-searches broadly
# (upgrading toward trusted sources) instead of chasing individual unverified
# claims, which tends to just re-find the same low-quality source.
LOW_SOURCE_CREDIBILITY_THRESHOLD: float = float(os.getenv("LOW_SOURCE_CREDIBILITY_THRESHOLD", "0.5"))

# Data Directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_SCRAPES_DIR = DATA_DIR / "raw_scrapes"
DB_DIR = DATA_DIR / "db"
DB_PATH = DB_DIR / "argus_checkpoints.db"
JOBS_DB_PATH = DB_DIR / "argus_jobs.db"
LOG_PATH = DATA_DIR / "logs" / "argus.log"
