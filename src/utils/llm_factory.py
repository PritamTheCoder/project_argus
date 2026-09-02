"""
Project Argus - LLM Factory

Provides a unified interface to instantiate LangChain chat models
across different providers (Gemini, Groq, OpenAI, NVIDIA).

Includes automatic fallback: if the requested provider fails or has an
invalid API key, falls back to Gemini as the guaranteed-available provider.
"""

import itertools
import logging
import os
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.rate_limiters import InMemoryRateLimiter

# Provider specific imports
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI

from src.config import (
    GEMINI_DEFAULT_MODEL, GEMINI_MAX_RETRIES, GEMINI_TIMEOUT,
    GEMINI_RPM, GROQ_RPM, OPENAI_RPM, NVIDIA_RPM, NEMOTRON_RPM, STEP_RPM, KIMI_RPM, GLM_RPM,
    RATE_LIMIT_BURST,
    NVIDIA_BASE_URL, NIM_TIMEOUT,
    NVIDIA_API_NEMOTRON3_KEY,
    STEP_API_KEY, KIMI_API_KEY, GLM_API_KEY,
)

logger = logging.getLogger(__name__)

# Fallback defaults (Gemini is the guaranteed working provider)
_FALLBACK_MODEL = GEMINI_DEFAULT_MODEL
_FALLBACK_PROVIDER = "gemini"

# ── Proactive client-side rate limiting ──────────────────────────────────────
# One shared limiter per provider, applied to every model we build. LangChain's
# InMemoryRateLimiter blocks (sync) / awaits (async) before each request so we
# stay under provider RPM limits proactively rather than reacting to 429s. It
# works transparently through .with_structured_output() and .bind_tools().
_PROVIDER_RPM = {
    "gemini": GEMINI_RPM,
    "groq": GROQ_RPM,
    "openai": OPENAI_RPM,
    "nvidia": NVIDIA_RPM,
    "nemotron": NEMOTRON_RPM,
    "step": STEP_RPM,
    "kimi": KIMI_RPM,
    "glm": GLM_RPM,
}
_rate_limiters: dict[str, InMemoryRateLimiter] = {}

# NVIDIA NIM providers reached via the OpenAI-compatible endpoint. Each maps to
# its own key + optional reasoning `extra_body`. Keys/extra_body are read at
# call time (via the helper below) so they stay patch-friendly for tests.
_NIM_PROVIDERS = ("nemotron", "step", "kimi", "glm")


def _nim_spec(provider: str) -> tuple[str, dict | None]:
    """Return (api_key, extra_body) for a NVIDIA NIM provider, read live."""
    if provider == "nemotron":
        # reasoning_budget removed: NVIDIA's model runner rejects it (400 on every call).
        return NVIDIA_API_NEMOTRON3_KEY, {
            "chat_template_kwargs": {"enable_thinking": True},
        }
    if provider == "step":
        return STEP_API_KEY, None  # multimodal chat; no reasoning extra_body
    if provider == "kimi":
        return KIMI_API_KEY, None  # plain large-context chat; no reasoning extra_body
    if provider == "glm":
        return GLM_API_KEY, None  # no reasoning extra_body; plain chat/structured-output mode
    return "", None


def _get_rate_limiter(name: str, rpm: int | None = None) -> InMemoryRateLimiter:
    """Shared limiter keyed by ``name``. ``rpm`` overrides the per-provider default
    (used for the per-key Groq limiters, whose name isn't a plain provider)."""
    if name not in _rate_limiters:
        r = rpm if rpm is not None else _PROVIDER_RPM.get(name, 12)
        _rate_limiters[name] = InMemoryRateLimiter(
            requests_per_second=max(r / 60.0, 0.05),
            check_every_n_seconds=0.1,
            max_bucket_size=RATE_LIMIT_BURST,  # burst, then smooth; ladder backstops 429s
        )
    return _rate_limiters[name]


# ── Groq multi-key pool ──────────────────────────────────────────────────────
# Round-robins across multiple Groq accounts (GROQ_API_KEY_A/_B/_C) to multiply
# free-tier RPM/TPD. Falls back to a single GROQ_API_KEY if none are set.
_groq_rr_counter = itertools.count()


def _groq_keys() -> list[str]:
    """Non-empty Groq keys in priority order. Read live so tests can patch env."""
    candidates = [
        os.getenv("GROQ_API_KEY_A", ""),
        os.getenv("GROQ_API_KEY_B", ""),
        os.getenv("GROQ_API_KEY_C", ""),
        os.getenv("GROQ_API_KEY", ""),  # legacy single-key fallback
    ]
    return [k.strip() for k in candidates if k.strip()]


def _next_groq_key() -> tuple[str, int]:
    """Round-robin the next (key, pool_index). Returns ('', -1) if none set."""
    keys = _groq_keys()
    if not keys:
        return "", -1
    idx = next(_groq_rr_counter) % len(keys)
    return keys[idx], idx


def _create_llm(model_name: str, provider: str, temperature: float) -> BaseChatModel:
    """
    Internal: create a single LLM instance for the given provider.
    Only passes provider-appropriate arguments. Every model is built with a
    shared per-provider rate limiter.
    """
    provider = provider.lower().strip()
    limiter = _get_rate_limiter(provider)

    if provider == "gemini":
        return ChatGoogleGenerativeAI(
            model=model_name,
            temperature=temperature,
            rate_limiter=limiter,
            max_retries=GEMINI_MAX_RETRIES,  # 1 = no in-SDK retries → fail fast to the fallback ladder
            timeout=GEMINI_TIMEOUT,          # seconds; hard ceiling on a single request
        )

    elif provider == "groq":
        # Round-robin across the account pool. Each key gets its OWN rate limiter
        # (its own RPM pool) so the two accounts' limits don't share a bucket.
        key, idx = _next_groq_key()
        groq_limiter = _get_rate_limiter(f"groq#{idx}", GROQ_RPM) if idx >= 0 else limiter
        if idx >= 0:
            logger.debug(f"Groq: using account-pool key #{idx}.")
        return ChatGroq(
            model_name=model_name,
            temperature=temperature,
            api_key=key or None,
            rate_limiter=groq_limiter,
        )

    elif provider == "openai":
        return ChatOpenAI(
            model=model_name,
            temperature=temperature,
            rate_limiter=limiter,
        )

    elif provider in _NIM_PROVIDERS:
        # NVIDIA NIM models (Nemotron / Step / Kimi / GLM) via the OpenAI-compatible
        # endpoint. Each has its own key (own quota) and optional reasoning params.
        api_key, extra_body = _nim_spec(provider)
        kwargs = dict(
            model=model_name,
            base_url=NVIDIA_BASE_URL,
            api_key=api_key,
            temperature=temperature,
            max_tokens=16384,
            rate_limiter=limiter,
            timeout=NIM_TIMEOUT,   # abort client-side before the gateway 504s (~5 min)
            max_retries=1,         # no in-SDK retries → fail fast to the fallback ladder
        )
        if extra_body:
            kwargs["extra_body"] = extra_body
        return ChatOpenAI(**kwargs)

    elif provider == "nvidia":
        from langchain_nvidia_ai_endpoints import ChatNVIDIA
        nvidia_key = os.getenv("NVIDIA_API_KEY", "")
        return ChatNVIDIA(
            model=model_name,
            api_key=nvidia_key,
            temperature=temperature,
            top_p=0.95,
            max_tokens=16384,
            rate_limiter=limiter,
        )

    else:
        raise ValueError(
            f"Unsupported LLM provider: '{provider}'. "
            "Supported: ['gemini', 'groq', 'openai', 'nvidia', 'nemotron', 'step', 'kimi', 'glm']"
        )


def get_llm(
    model_name: str,
    provider: str,
    temperature: float = 0.0,
    **kwargs  # Accepted for backward compat but not forwarded (prevents cross-provider kwarg issues)
) -> BaseChatModel:
    """
    Instantiate the appropriate LangChain BaseChatModel.

    If the requested provider has an invalid/missing API key, automatically
    falls back to the default Gemini model so the pipeline never hard-crashes.

    Args:
        model_name: The specific model ID (e.g., "gemini-2.5-flash", "llama-3.3-70b-versatile").
        provider: The provider name ("gemini", "groq", "openai").
        temperature: The sampling temperature.

    Returns:
        BaseChatModel: An instantiated LangChain chat model.
    """
    provider_lower = provider.lower().strip()

    if provider_lower == "groq":
        groq_keys = _groq_keys()
        if not groq_keys:
            logger.warning(
                f"No Groq API key set (GROQ_API_KEY_A/_B/_C or GROQ_API_KEY). "
                f"Falling back to {_FALLBACK_PROVIDER}/{_FALLBACK_MODEL}."
            )
            return _create_llm(_FALLBACK_MODEL, _FALLBACK_PROVIDER, temperature)
        if not any(k.startswith("gsk_") for k in groq_keys):
            logger.warning(
                f"No Groq key looks valid (expected 'gsk_...' prefix). "
                f"Falling back to {_FALLBACK_PROVIDER}/{_FALLBACK_MODEL}."
            )
            return _create_llm(_FALLBACK_MODEL, _FALLBACK_PROVIDER, temperature)

    nvidia_key = os.getenv("NVIDIA_API_KEY", "")
    if provider_lower == "nvidia" and not nvidia_key:
        logger.warning(f"NVIDIA_API_KEY is not set. Falling back to {_FALLBACK_PROVIDER}/{_FALLBACK_MODEL}.")
        return _create_llm(_FALLBACK_MODEL, _FALLBACK_PROVIDER, temperature)

    if provider_lower in _NIM_PROVIDERS:
        nim_key, _ = _nim_spec(provider_lower)
        if not nim_key:
            logger.warning(
                f"NVIDIA NIM key for provider '{provider_lower}' is not set. "
                f"Falling back to {_FALLBACK_PROVIDER}/{_FALLBACK_MODEL}."
            )
            return _create_llm(_FALLBACK_MODEL, _FALLBACK_PROVIDER, temperature)

    # Attempt primary provider
    try:
        llm = _create_llm(model_name, provider_lower, temperature)
        logger.info(f"LLM Factory: Initialized {model_name} via {provider_lower}")
        return llm
    except Exception as e:
        if provider_lower == _FALLBACK_PROVIDER:
            logger.error(f"Failed to initialize LLM {model_name} from {provider_lower}: {e}")
            raise

        logger.warning(
            f"Failed to initialize {model_name} via {provider_lower}: {e}. "
            f"Falling back to {_FALLBACK_PROVIDER}/{_FALLBACK_MODEL}."
        )
        return _create_llm(_FALLBACK_MODEL, _FALLBACK_PROVIDER, temperature)


def parse_fallback_chain(chain: str) -> list[tuple[str, str]]:
    """Parse a "model:provider,model:provider" string into (model, provider) tuples.

    Malformed entries are skipped with a warning so a bad env var degrades to
    "no fallback" rather than crashing the pipeline.
    """
    specs: list[tuple[str, str]] = []
    for entry in (chain or "").split(","):
        entry = entry.strip()
        if not entry:
            continue
        # rsplit so model names that themselves contain ':' survive (rare, but safe).
        model, sep, provider = entry.rpartition(":")
        if not sep or not model or not provider:
            logger.warning(f"Ignoring malformed fallback entry '{entry}' (expected 'model:provider').")
            continue
        specs.append((model.strip(), provider.strip()))
    return specs


def get_llm_with_fallbacks(
    model_name: str,
    provider: str,
    fallback_chain: str = "",
    temperature: float = 0.0,
    structured_schema=None,
    tools=None,
):
    """Build a primary LLM with a *runtime*, cross-provider fallback ladder.

    Unlike ``get_llm`` (whose fallback only triggers at construction time), this
    uses LangChain's ``.with_fallbacks()`` so a runtime error on ``.invoke()`` —
    notably a 429/503 from an exhausted or overloaded provider — fails over to the
    next provider in ``fallback_chain``. The wrapping applies to every rung so the
    chain behaves identically regardless of which one fires.

    Args:
        model_name/provider: the primary model.
        fallback_chain: "model:provider,model:provider" tried in order on failure.
        structured_schema: if set, every rung is wrapped with structured output.
        tools: if set, every rung is bound with these tools (``.bind_tools``).
            Mutually exclusive with ``structured_schema``.
    """
    if structured_schema is not None and tools is not None:
        raise ValueError("Pass only one of structured_schema / tools.")

    def _build(m: str, p: str):
        llm = get_llm(m, p, temperature)
        if structured_schema is not None:
            return llm.with_structured_output(structured_schema)
        if tools is not None:
            return llm.bind_tools(tools)
        return llm

    primary = _build(model_name, provider)
    fallbacks = []
    for m, p in parse_fallback_chain(fallback_chain):
        try:
            fallbacks.append(_build(m, p))
        except Exception as e:
            logger.warning(f"Skipping fallback {m}/{p}: failed to build ({e}).")

    if not fallbacks:
        return primary
    logger.info(
        f"LLM Factory: {model_name}/{provider} armed with {len(fallbacks)} "
        f"cross-provider fallback(s)."
    )
    return primary.with_fallbacks(fallbacks)
