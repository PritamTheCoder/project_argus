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
# NVIDIA NIM (OpenAI-compatible) endpoint + per-model keys. Each model has its
# own key, i.e. its own free-tier quota — assigning agents across them spreads
# rate-limit load. All are reached via the same OpenAI-compatible base URL.
NVIDIA_BASE_URL: str = os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")

# Nemotron-3 Ultra (reasoning).
NVIDIA_API_NEMOTRON3_KEY: str = os.getenv("NVIDIA_API_NEMOTRON3_KEY", "")
NEMOTRON_MODEL: str = os.getenv("NEMOTRON_MODEL", "nvidia/nemotron-3-ultra-550b-a55b")
NEMOTRON_REASONING_BUDGET: int = int(os.getenv("NEMOTRON_REASONING_BUDGET", "4096"))

# StepFun Step-3.7 Flash (multimodal / vision-capable chat). Live-verified
# working 2026-08-18 (an earlier 401 on this key has since resolved itself).
STEP_API_KEY: str = os.getenv("STEP_3.7_API_KEY", "")
STEP_MODEL: str = os.getenv("STEP_MODEL", "stepfun-ai/step-3.7-flash")

# Kimi K2.6 (Moonshot, large-context non-reasoning chat) via NVIDIA NIM.
# NOT used in any default below as of 2026-08-18: this account's key gets a 404
# ("Function ... Not found for account") specifically for kimi-k2.6 — the model
# ID itself is correct (verified against NVIDIA's own NIM reference docs), but
# it isn't provisioned/entitled for this NVIDIA developer account. Request
# access for the model on build.nvidia.com to re-enable it, then add it back
# into the relevant *_FALLBACK_CHAIN env var(s) below.
KIMI_API_KEY: str = os.getenv("KIMI_NVIDIA_KEY", "")
KIMI_MODEL: str = os.getenv("KIMI_MODEL", "moonshotai/kimi-k2.6")

# Z.AI GLM-5.2 (Zhipu AI) via NVIDIA NIM. Large context (131K), full tool-calling
# + structured-output support, generous free NIM quota. Live-verified 2026-08-18
# against this account — the two prior GLM generations (glm-5.1, glm-4.7) both
# returned HTTP 410 "reached end of life" on the same check, so this model ID
# WILL need re-verifying again eventually; don't assume it's permanent.
GLM_API_KEY: str = os.getenv("ZLM_NVIDIA_KEY", "")
GLM_MODEL: str = os.getenv("GLM_MODEL", "z-ai/glm-5.2")

# ── Default Gemini model ─────────────────────────────────────────────────────
# gemini-2.5-flash-lite is deprecated and shuts down 2026-10-16 (was previously
# the default here). Live-verified 2026-08-18: gemini-3.1-flash-lite is the
# current GA (non-preview) flash-lite model on this key's free tier — Google's
# own migration guidance names 3.1-flash-lite as the successor tier. Re-check
# against `GET generativelanguage.googleapis.com/v1beta/models` if this starts
# 404ing; Google rotates these every few months.
GEMINI_DEFAULT_MODEL: str = os.getenv("GEMINI_DEFAULT_MODEL", "gemini-3.1-flash-lite")

# Agent LLM Settings.
# Shared cross-provider fallback ladder for the Groq-primary structured nodes:
# GLM (own NIM quota) then Gemini (emergency). Keeps a single provider's daily
# token cap from ERRORing a whole run — the failure mode that broke the eval.
_STRUCTURED_FALLBACK = f"{GLM_MODEL}:glm,{GEMINI_DEFAULT_MODEL}:gemini"

# llama-3.3-70b-versatile (the previous default here) was deprecated by Groq on
# 2026-06-17 and has been removed from this account's model list entirely (a
# real, confirmed-live outage — see error.md). openai/gpt-oss-120b is Groq's own
# documented migration target: same free-tier RPM (30), native tool-calling and
# structured-output support (it's OpenAI's own open-weight release, built for
# exactly this).
LIBRARIAN_MODEL: str = os.getenv("LIBRARIAN_MODEL", "openai/gpt-oss-120b")
LIBRARIAN_PROVIDER: str = os.getenv("LIBRARIAN_PROVIDER", "groq")
LIBRARIAN_FALLBACK_CHAIN: str = os.getenv("LIBRARIAN_FALLBACK_CHAIN", _STRUCTURED_FALLBACK)

CRITIC_MODEL: str    = os.getenv("CRITIC_MODEL", "openai/gpt-oss-120b")
CRITIC_PROVIDER: str = os.getenv("CRITIC_PROVIDER", "groq")
CRITIC_FALLBACK_CHAIN: str = os.getenv("CRITIC_FALLBACK_CHAIN", _STRUCTURED_FALLBACK)

# Verifier: high-frequency, structured-output, small batches (≤30 facts). Primary
# is Groq gpt-oss-120b — generous free RPM and fits the batch size. Gemini's free
# tier is too small to be a primary here; it's demoted to the last fallback rung
# where its quota is fine as an emergency.
VERIFIER_MODEL: str  = os.getenv("VERIFIER_MODEL", "openai/gpt-oss-120b")
VERIFIER_PROVIDER: str = os.getenv("VERIFIER_PROVIDER", "groq")
# Cross-provider runtime fallback ladder. When the primary fails at *call* time
# (429/503/timeout) the request fails over to the NEXT entry — provider diversity
# is the only real mitigation. Format: comma-separated "model:provider"; all do
# structured output and sit on independent capacity pools.
VERIFIER_FALLBACK_CHAIN: str = os.getenv("VERIFIER_FALLBACK_CHAIN", _STRUCTURED_FALLBACK)

# Writer: long-form synthesis (no structured output). Default to the Nemotron
# reasoning model when its key is present (high quality + offloads Gemini),
# otherwise fall back to the default Gemini model. The Writer guards its call
# with an in-node fallback so a Nemotron error never blocks a report.
_writer_default_model = NEMOTRON_MODEL if NVIDIA_API_NEMOTRON3_KEY else GEMINI_DEFAULT_MODEL
_writer_default_provider = "nemotron" if NVIDIA_API_NEMOTRON3_KEY else "gemini"
WRITER_MODEL: str    = os.getenv("WRITER_MODEL", _writer_default_model)
WRITER_PROVIDER: str = os.getenv("WRITER_PROVIDER", _writer_default_provider)
# Writer is plain long-form chat (no structured output); GLM then Gemini are both
# fine for synthesis and sit on quota independent of Nemotron.
WRITER_FALLBACK_CHAIN: str = os.getenv("WRITER_FALLBACK_CHAIN", _STRUCTURED_FALLBACK)

# Embedding Settings
EMBEDDING_MODE: str = os.getenv("EMBEDDING_MODE", "local") # "local" or "openai"
LOCAL_EMBEDDING_MODEL: str = os.getenv("LOCAL_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
MAX_CHUNK_TOKENS: int = int(os.getenv("MAX_CHUNK_TOKENS", "300"))
TOP_K_CHUNKS: int = int(os.getenv("TOP_K_CHUNKS", "15"))

# Refiner Settings
# Refiner sends entire scraped documents (~40K+ tokens), so the model MUST be
# large-context. Kimi-K2.6 was the intended primary here (large context, plain
# completions) but is not usable on this account (see KIMI_API_KEY note above),
# so GLM-5.2 is primary instead (131K context, own NIM quota, structured output
# confirmed). Gemini's free tier can't sustain a high-frequency node as primary.
REFINER_MODEL: str = os.getenv("REFINER_MODEL", GLM_MODEL)
REFINER_PROVIDER: str = os.getenv("REFINER_PROVIDER", "glm")
# Cross-provider runtime fallback ladder for the refiner. A runtime failure on the
# primary fails over to the NEXT entry; every rung must be large-context.
REFINER_FALLBACK_CHAIN: str = os.getenv(
    "REFINER_FALLBACK_CHAIN",
    f"openai/gpt-oss-120b:groq,{GEMINI_DEFAULT_MODEL}:gemini",
)

# ── Client-side rate limiting ────────────────────────────────────────────────
# Proactive per-provider throttle (requests/minute) applied to every LLM call
# so we stay under provider free-tier RPM limits instead of reactively eating
# 429s. Tune per your tier. Gemini free flash-lite ≈ 30 RPM → keep margin.
# google-genai "attempts" (total tries incl. the first); 0 or 1 means NO retries.
# We want NO in-SDK retries: on a 429/503 the SDK otherwise sleeps *inside* the
# call (honouring the server's RetryInfo, which on daily-quota exhaustion can be
# minutes) before raising — which stalls the run and starves the cross-provider
# fallback ladder, since `.with_fallbacks()` only fires once the primary RAISES.
# Failing fast hands off to an independent provider in ~1s. The ladder IS the retry.
GEMINI_MAX_RETRIES: int = int(os.getenv("GEMINI_MAX_RETRIES", "1"))
# Hard per-request ceiling (seconds) so a hung socket can't block the run either.
GEMINI_TIMEOUT: int = int(os.getenv("GEMINI_TIMEOUT", "60"))
GEMINI_RPM: int = int(os.getenv("GEMINI_RPM", "25"))
GROQ_RPM: int = int(os.getenv("GROQ_RPM", "30"))  # openai/gpt-oss-120b free-tier ceiling; x2 with the key pool
OPENAI_RPM: int = int(os.getenv("OPENAI_RPM", "60"))
NVIDIA_RPM: int = int(os.getenv("NVIDIA_RPM", "30"))
# NVIDIA NIM reasoning models are large; keep their rates conservative.
NEMOTRON_RPM: int = int(os.getenv("NEMOTRON_RPM", "8"))
STEP_RPM: int = int(os.getenv("STEP_RPM", "10"))
KIMI_RPM: int = int(os.getenv("KIMI_RPM", "10"))  # unused by default — see KIMI_API_KEY note
# GLM-5.1 was reported free on NIM at up to 40 RPM with no daily cap; staying
# under that with margin since GLM-5.2's own limit isn't independently confirmed.
GLM_RPM: int = int(os.getenv("GLM_RPM", "20"))
# Burst size for the proactive limiter: how many calls fire immediately before it
# throttles to the steady RPM. Raised now that every node has a cross-provider
# fallback ladder — parallel bursts (see the Scout) flow through instead of being
# serialized, and an occasional 429 fails over rather than stalling the run.
RATE_LIMIT_BURST: int = int(os.getenv("RATE_LIMIT_BURST", "8"))
# Hard per-request ceiling (seconds) for NIM (OpenAI-compatible) calls. NVIDIA's
# gateway can sit on a slow request for ~5 min before a 504; this aborts client-side
# first so a stuck NIM call fails over to the next fallback rung quickly.
NIM_TIMEOUT: int = int(os.getenv("NIM_TIMEOUT", "120"))

# Search Settings
MAX_SEARCH_RESULTS: int = int(os.getenv("MAX_SEARCH_RESULTS", "5"))

# ── Tool layer / multi-backend search ───────────────────────────────────────
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

# Tool search: when the tool catalog exceeds this many tools, the acquisition
# agent binds only the top-N most relevant tools rather than every tool. Below
# this, all tools are bound (no embedding work).
TOOL_SEARCH_MAX_TOOLS: int = int(os.getenv("TOOL_SEARCH_MAX_TOOLS", "6"))

# MCP (Model Context Protocol) servers to load tools from at startup. JSON map
# of {server_name: {transport, command/args or url}}. Empty = MCP disabled.
# Example: {"fs": {"transport": "stdio", "command": "npx",
#                  "args": ["-y", "@modelcontextprotocol/server-filesystem", "/data"]}}
MCP_SERVERS_JSON: str = os.getenv("MCP_SERVERS_JSON", "")

# Acquisition agent (the tool-calling source gatherer). Must be a model that
# supports tool/function calling. Defaults to Groq (gpt-oss-120b, tool-calling
# native): the gatherer fires several tool-calling rounds per sub-query, so
# keeping it off Gemini keeps Gemini RPM usage low. A deterministic fallback
# covers any tool-calling failure.
GATHERER_MODEL: str = os.getenv("GATHERER_MODEL", "openai/gpt-oss-120b")
GATHERER_PROVIDER: str = os.getenv("GATHERER_PROVIDER", "groq")
# Tool-calling fallback ladder (GLM + Gemini both do tool calling). Applies on
# top of the deterministic mode-aware fallback the gatherer already has.
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
    "harvard.edu",
    # Note: .gov and .edu top-level domains will be handled via regex/endswith in the filter logic
}

# Data Directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_SCRAPES_DIR = DATA_DIR / "raw_scrapes"
DB_DIR = DATA_DIR / "db"
DB_PATH = DB_DIR / "argus_checkpoints.db"
JOBS_DB_PATH = DB_DIR / "argus_jobs.db"
LOG_PATH = DATA_DIR / "logs" / "argus.log"
