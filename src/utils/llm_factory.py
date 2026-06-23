"""
Project Argus - LLM Factory

Provides a unified interface to instantiate LangChain chat models
across different providers (Gemini, Groq, OpenAI, NVIDIA).

Includes automatic fallback: if the requested provider fails or has an
invalid API key, falls back to Gemini as the guaranteed-available provider.
"""

import logging
import os
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.rate_limiters import InMemoryRateLimiter

# Provider specific imports
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI

from src.config import (
    GEMINI_DEFAULT_MODEL,
    GEMINI_RPM, GROQ_RPM, OPENAI_RPM, NVIDIA_RPM, NEMOTRON_RPM, DEEPSEEK_RPM, STEP_RPM,
    NVIDIA_BASE_URL,
    NVIDIA_API_NEMOTRON3_KEY, NEMOTRON_REASONING_BUDGET,
    DEEPSEEK_API_KEY, STEP_API_KEY,
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
    "deepseek": DEEPSEEK_RPM,
    "step": STEP_RPM,
}
_rate_limiters: dict[str, InMemoryRateLimiter] = {}

# NVIDIA NIM providers reached via the OpenAI-compatible endpoint. Each maps to
# its own key + optional reasoning `extra_body`. Keys/extra_body are read at
# call time (via the helper below) so they stay patch-friendly for tests.
_NIM_PROVIDERS = ("nemotron", "deepseek", "step")


def _nim_spec(provider: str) -> tuple[str, dict | None]:
    """Return (api_key, extra_body) for a NVIDIA NIM provider, read live."""
    if provider == "nemotron":
        return NVIDIA_API_NEMOTRON3_KEY, {
            "chat_template_kwargs": {"enable_thinking": True},
            "reasoning_budget": NEMOTRON_REASONING_BUDGET,
        }
    if provider == "deepseek":
        return DEEPSEEK_API_KEY, {
            "chat_template_kwargs": {"thinking": True, "reasoning_effort": "high"},
        }
    if provider == "step":
        return STEP_API_KEY, None  # multimodal chat; no reasoning extra_body
    return "", None


def _get_rate_limiter(provider: str) -> InMemoryRateLimiter:
    if provider not in _rate_limiters:
        rpm = _PROVIDER_RPM.get(provider, 12)
        _rate_limiters[provider] = InMemoryRateLimiter(
            requests_per_second=max(rpm / 60.0, 0.05),
            check_every_n_seconds=0.1,
            max_bucket_size=3,  # allow a tiny burst, then smooth
        )
    return _rate_limiters[provider]


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
        )

    elif provider == "groq":
        return ChatGroq(
            model_name=model_name,
            temperature=temperature,
            rate_limiter=limiter,
        )

    elif provider == "openai":
        return ChatOpenAI(
            model=model_name,
            temperature=temperature,
            rate_limiter=limiter,
        )

    elif provider in _NIM_PROVIDERS:
        # NVIDIA NIM models (Nemotron / DeepSeek / Step) via the OpenAI-compatible
        # endpoint. Each has its own key (own quota) and optional reasoning params.
        api_key, extra_body = _nim_spec(provider)
        kwargs = dict(
            model=model_name,
            base_url=NVIDIA_BASE_URL,
            api_key=api_key,
            temperature=temperature,
            max_tokens=16384,
            rate_limiter=limiter,
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
            "Supported: ['gemini', 'groq', 'openai', 'nvidia', 'nemotron', 'deepseek', 'step']"
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

    # Key prefix validation (catches common misconfigurations)
    groq_key = os.getenv("GROQ_API_KEY", "")
    if provider_lower == "groq" and groq_key and not groq_key.startswith("gsk_"):
        logger.warning(
            f"GROQ_API_KEY doesn't look like a valid Groq key (expected 'gsk_...' prefix). "
            f"Falling back to {_FALLBACK_PROVIDER}/{_FALLBACK_MODEL}."
        )
        return _create_llm(_FALLBACK_MODEL, _FALLBACK_PROVIDER, temperature)

    if provider_lower == "groq" and not groq_key:
        logger.warning(f"GROQ_API_KEY is not set. Falling back to {_FALLBACK_PROVIDER}/{_FALLBACK_MODEL}.")
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
