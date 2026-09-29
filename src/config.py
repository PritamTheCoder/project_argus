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

# Nemotron-3 Ultra (reasoning). Not on the default Writer ladder — a slow
# 503 failover blows the per-run timeout. Set WRITER_FALLBACK_CHAIN to use it.
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
# Nemotron is deliberately NOT on this ladder: it fails over slowly enough
# (long retry-after on a 503) to blow the per-case timeout on its own. Set
# WRITER_FALLBACK_CHAIN to re-insert it if its availability improves.
WRITER_FALLBACK_CHAIN: str = os.getenv("WRITER_FALLBACK_CHAIN", _STRUCTURED_FALLBACK)

# Citation auditor: checks the Writer's own sentences against their cited
# evidence (Phase 10.A) — a structured judgment call like Verifier/Critic,
# not prose generation, so it shares their model tier rather than the Writer's.
CITATION_AUDITOR_MODEL: str = os.getenv("CITATION_AUDITOR_MODEL", "openai/gpt-oss-120b")
CITATION_AUDITOR_PROVIDER: str = os.getenv("CITATION_AUDITOR_PROVIDER", "groq")
CITATION_AUDITOR_FALLBACK_CHAIN: str = os.getenv("CITATION_AUDITOR_FALLBACK_CHAIN", _STRUCTURED_FALLBACK)
# Sentences per LLM call — batched so a long report doesn't cost one call per sentence.
CITATION_AUDIT_BATCH_SIZE: int = int(os.getenv("CITATION_AUDIT_BATCH_SIZE", "15"))

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
# budget, NOT its context window. Free-tier Groq allows ~8K tokens/minute for
# this model, so a batch near that ceiling consumes the whole minute and 413s
# the moment any other call overlaps it. ~4 chars/token, so 12K chars ≈ 3K
# tokens of documents, leaving room for the schema, instructions, the response,
# and the other nodes sharing the same bucket.
REFINER_BATCH_MAX_CHARS: int = int(os.getenv("REFINER_BATCH_MAX_CHARS", "12000"))
# Ceiling on a single document's contribution, so one huge page can't by itself
# produce a batch that breaches the budget.
REFINER_MAX_DOC_CHARS: int = int(os.getenv("REFINER_MAX_DOC_CHARS", "12000"))

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

# Tokens-per-minute ceilings. RPM alone is the wrong constraint: free-tier Groq
# rejects on tokens long before requests. 0 disables the token gate for that provider.
GROQ_TPM: int = int(os.getenv("GROQ_TPM", "8000"))       # per key, like GROQ_RPM
GEMINI_TPM: int = int(os.getenv("GEMINI_TPM", "0"))      # free tier gates on RPM/RPD, not TPM
OPENAI_TPM: int = int(os.getenv("OPENAI_TPM", "0"))
NVIDIA_TPM: int = int(os.getenv("NVIDIA_TPM", "0"))
NEMOTRON_TPM: int = int(os.getenv("NEMOTRON_TPM", "0"))
# Tokens assumed for a call before its true size is known. The limiter replaces
# this with a running average of observed usage as responses come back.
RATE_LIMIT_DEFAULT_CALL_TOKENS: int = int(os.getenv("RATE_LIMIT_DEFAULT_CALL_TOKENS", "2000"))
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
# Semantic Scholar API key (optional). Unauthenticated requests share one global
# pool and get throttled during heavy use; a key gives a dedicated 1 RPS.
# SS2_API_KEY is accepted as an alias.
SEMANTIC_SCHOLAR_API_KEY: str = os.getenv("SEMANTIC_SCHOLAR_API_KEY") or os.getenv("SS2_API_KEY", "")
# Client-side ceiling. The Scout issues sub-queries concurrently, so without
# this a burst would exceed the 1 RPS allowance and be rejected.
SEMANTIC_SCHOLAR_RPS: float = float(os.getenv("SEMANTIC_SCHOLAR_RPS", "1.0"))
# Repeat sub-queries across loop iterations are served from cache, not quota.
SEMANTIC_SCHOLAR_CACHE_TTL_HOURS: int = int(os.getenv("SEMANTIC_SCHOLAR_CACHE_TTL_HOURS", "24"))
# Crossref "polite pool" contact (recommended by Crossref ToS).
CROSSREF_MAILTO: str = os.getenv("CROSSREF_MAILTO", "research@project-argus.local")
# Shared HTTP timeout (seconds) for tool/provider network calls.
TOOL_HTTP_TIMEOUT: int = int(os.getenv("TOOL_HTTP_TIMEOUT", "20"))
# Default result count for academic backends.
ACADEMIC_MAX_RESULTS: int = int(os.getenv("ACADEMIC_MAX_RESULTS", "8"))

# Tool search: when the tool catalog exceeds this many tools, the acquisition
# agent binds only the top-N most relevant tools rather than every tool. Below
# this, all tools are bound (no embedding work).
TOOL_SEARCH_MAX_TOOLS: int = int(os.getenv("TOOL_SEARCH_MAX_TOOLS", "8"))

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
# kg_lookup scope. Default "global": the memory tool and Librarian's pre-flight
# check reuse prior runs' verified facts, scoped to the caller's own
# owner_key_hash (the KG has no other tenant boundary). Set "session" to fall
# back to this-run-only lookups, consistent with the session-scoped
# Critic/Reflector retrieval.
KG_LOOKUP_GLOBAL: bool = os.getenv("KG_LOOKUP_SCOPE", "global").lower() == "global"

# Scraper Settings
SCRAPE_TIMEOUT: int = int(os.getenv("SCRAPE_TIMEOUT", "15"))
# Seconds to let a thin page finish rendering before re-reading it. Cheaper and
# more effective than relaunching a browser, which is what this replaced.
SCRAPE_RETRY_SETTLE_S: float = float(os.getenv("SCRAPE_RETRY_SETTLE_S", "2.0"))
# Retry thin pages in a visible browser. Off by default: launching a second,
# headed Chromium per thin page is pure thermal/CPU cost with no extraction
# benefit for most sites. Enable only for sites that detect headless browsers.
SCRAPE_RETRY_HEADED: bool = os.getenv("SCRAPE_RETRY_HEADED", "false").lower() == "true"
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

# Query-to-source relevance. Credibility says a source is trustworthy; it says
# nothing about whether the source is about the question, and retrieval only
# ranks candidates relative to each other. Below this cosine similarity a
# source is logged as likely off-topic however authoritative it scores.
LOW_RELEVANCE_THRESHOLD: float = float(os.getenv("LOW_RELEVANCE_THRESHOLD", "0.35"))
# Drop off-topic sources instead of only logging them. Credibility scores what
# a URL *looks* like, not what it's about, so an off-topic source can still rate
# highly and reach (and be charged for by) the Refiner. Set false to measure
# without filtering.
RELEVANCE_GATE_ENABLED: bool = os.getenv("RELEVANCE_GATE_ENABLED", "true").lower() == "true"
# Floor, not a target: restoring a known off-topic source is the very harm the
# gate exists to prevent, so it happens only when the alternative is no evidence
# at all. Thin-but-relevant evidence is the Critic's job to re-search, not this
# backstop's job to pad.
RELEVANCE_GATE_MIN_SOURCES: int = int(os.getenv("RELEVANCE_GATE_MIN_SOURCES", "1"))
# Characters of a source sampled for the relevance embedding.
RELEVANCE_SAMPLE_CHARS: int = int(os.getenv("RELEVANCE_SAMPLE_CHARS", "2000"))

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
