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
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.language_models.chat_models import BaseChatModel
from src.utils.rate_limit import TokenAwareRateLimiter, TokenUsageReporter

# Provider specific imports
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI

from src.config import (
    GEMINI_DEFAULT_MODEL, GEMINI_MAX_RETRIES, GEMINI_TIMEOUT,
    GEMINI_RPM, GROQ_RPM, OPENAI_RPM, NVIDIA_RPM, NEMOTRON_RPM, STEP_RPM, KIMI_RPM, GLM_RPM,
    GEMINI_TPM, GROQ_TPM, OPENAI_TPM, NVIDIA_TPM, NEMOTRON_TPM,
    RATE_LIMIT_DEFAULT_CALL_TOKENS,
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
# One shared limiter per provider, applied to every model we build. It blocks
# (sync) / awaits (async) before each request so we stay under provider limits
# proactively rather than reacting to 429s, and works transparently through
# .with_structured_output() and .bind_tools().
# Both requests/minute AND tokens/minute are gated: Groq's free tier rejects on
# tokens first, so RPM alone let large prompts through and ate 429s.
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
# Tokens/minute per provider; 0 = no token ceiling, gate on requests only.
_PROVIDER_TPM = {
    "gemini": GEMINI_TPM,
    "groq": GROQ_TPM,
    "openai": OPENAI_TPM,
    "nvidia": NVIDIA_TPM,
    "nemotron": NEMOTRON_TPM,
}
_rate_limiters: dict[str, TokenAwareRateLimiter] = {}

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


def _get_rate_limiter(
    name: str, rpm: int | None = None, tpm: int | None = None
) -> TokenAwareRateLimiter:
    """Shared limiter keyed by ``name``. ``rpm``/``tpm`` override the per-provider
    defaults (used for the per-key Groq limiters, whose name isn't a plain
    provider)."""
    if name not in _rate_limiters:
        r = rpm if rpm is not None else _PROVIDER_RPM.get(name, 12)
        t = tpm if tpm is not None else _PROVIDER_TPM.get(name, 0)
        _rate_limiters[name] = TokenAwareRateLimiter(
            requests_per_minute=r,
            tokens_per_minute=t,
            burst=RATE_LIMIT_BURST,  # burst, then smooth; ladder backstops 429s
            default_call_tokens=RATE_LIMIT_DEFAULT_CALL_TOKENS,
        )
    return _rate_limiters[name]


class RungFailureLogger(BaseCallbackHandler):
    """Logs a ladder rung's real error even when a later rung masks it —
    otherwise only a bare status line (e.g. "400 Bad Request") ever surfaces."""

    def on_llm_error(self, error: BaseException, **kwargs) -> None:
        logger.warning("LLM rung failed: %s", error)


def _default_callbacks(limiter: TokenAwareRateLimiter) -> list:
    """Callbacks attached to every LLM instance this factory builds."""
    callbacks = [RungFailureLogger()]
    if limiter.gates_on_tokens:
        callbacks.append(TokenUsageReporter(limiter))
    return callbacks


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


def _groq_key_at(index: int) -> tuple[str, int]:
    """The pool key at ``index``, wrapped. Returns ('', -1) if the pool is empty."""
    keys = _groq_keys()
    if not keys:
        return "", -1
    idx = index % len(keys)
    return keys[idx], idx


def _create_llm(
    model_name: str, provider: str, temperature: float,
    groq_key_index: int | None = None,
) -> BaseChatModel:
    """
    Internal: create a single LLM instance for the given provider.
    Only passes provider-appropriate arguments. Every model is built with a
    shared per-provider rate limiter.

    ``groq_key_index`` pins this instance to one account in the Groq pool.
    Default (None) round-robins, which spreads load; the fallback ladder pins
    each rung instead, so a key that 429s hands off to a *different account*
    rather than to a different provider.
    """
    provider = provider.lower().strip()
    limiter = _get_rate_limiter(provider)

    if provider == "gemini":
        return ChatGoogleGenerativeAI(
            model=model_name,
            temperature=temperature,
            rate_limiter=limiter,
            callbacks=_default_callbacks(limiter),
            max_retries=GEMINI_MAX_RETRIES,  # 1 = no in-SDK retries → fail fast to the fallback ladder
            timeout=GEMINI_TIMEOUT,          # seconds; hard ceiling on a single request
        )

    elif provider == "groq":
        # Round-robin across the account pool. Each key gets its OWN rate limiter
        # (its own RPM pool) so the two accounts' limits don't share a bucket.
        key, idx = (
            _next_groq_key() if groq_key_index is None else _groq_key_at(groq_key_index)
        )
        groq_limiter = _get_rate_limiter(f"groq#{idx}", GROQ_RPM, GROQ_TPM) if idx >= 0 else limiter
        if idx >= 0:
            logger.debug(f"Groq: using account-pool key #{idx}.")
        return ChatGroq(
            model_name=model_name,
            temperature=temperature,
            api_key=key or None,
            rate_limiter=groq_limiter,
            callbacks=_default_callbacks(groq_limiter),
            # No in-SDK retries: the client honours retry-after and would sleep
            # re-asking a quota-exhausted key. The next rung is a different
            # account, so failing over immediately is faster and likelier to work.
            max_retries=0,
        )

    elif provider == "openai":
        return ChatOpenAI(
            model=model_name,
            temperature=temperature,
            rate_limiter=limiter,
            callbacks=_default_callbacks(limiter),
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
            callbacks=_default_callbacks(limiter),
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
            callbacks=_default_callbacks(limiter),
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
    groq_key_index: int | None = None,
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
        llm = _create_llm(model_name, provider_lower, temperature, groq_key_index)
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


def _structured(llm: BaseChatModel, schema):
    """Wrap a model for structured output using constrained decoding.

    Prefers ``method="json_schema"``: the grammar constrains generation, so the
    shape is guaranteed. The default on some providers is ``function_calling``,
    where a wrapper schema like ``{results: [...]}`` invites the model to emit a
    bare array instead of calling the tool — the request is then rejected with
    `tool_use_failed` even though the content was correct.

    Falls back to the provider default if it doesn't accept the argument, so an
    unfamiliar provider degrades rather than failing to build.
    """
    try:
        return llm.with_structured_output(schema, method="json_schema")
    except Exception as e:  # noqa: BLE001
        logger.debug("json_schema structured output unavailable (%s); using provider default.", e)
        return llm.with_structured_output(schema)


def _groq_pool_rungs(model_name: str, provider: str) -> list[tuple[str, str, int | None]]:
    """Ladder rungs for the primary model, one per Groq account when it has a pool.

    Groq's daily token budget (TPD) is per *organization*, so separate accounts
    hold separate budgets — a 429 on one should try its siblings before falling
    over to another provider. The first rung follows the round-robin so load
    still spreads across accounts; the rest are its siblings in order.
    """
    if provider.lower().strip() != "groq":
        return [(model_name, provider, None)]
    pool_size = len(_groq_keys())
    if pool_size <= 1:
        return [(model_name, provider, None)]
    _, start = _next_groq_key()
    return [(model_name, provider, (start + i) % pool_size) for i in range(pool_size)]


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

    def _build(m: str, p: str, groq_key_index: int | None = None):
        llm = get_llm(m, p, temperature, groq_key_index=groq_key_index)
        if structured_schema is not None:
            return _structured(llm, structured_schema)
        if tools is not None:
            return llm.bind_tools(tools)
        return llm

    # Rungs to try in order, as (model, provider, groq_key_index).
    rungs = _groq_pool_rungs(model_name, provider)
    rungs += [(m, p, None) for m, p in parse_fallback_chain(fallback_chain)]

    primary = _build(*rungs[0])
    fallbacks = []
    for m, p, key_index in rungs[1:]:
        try:
            fallbacks.append(_build(m, p, key_index))
        except Exception as e:
            logger.warning(f"Skipping fallback {m}/{p}: failed to build ({e}).")

    if not fallbacks:
        return primary
    sibling_keys = sum(1 for _, _, k in rungs[1:] if k is not None)
    logger.info(
        f"LLM Factory: {model_name}/{provider} armed with {len(fallbacks)} "
        f"fallback(s) ({sibling_keys} sibling Groq account(s))."
    )
    return primary.with_fallbacks(fallbacks)
