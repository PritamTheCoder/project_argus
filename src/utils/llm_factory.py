"""
Project Argus - LLM Factory

Provides a unified interface to instantiate LangChain chat models
across different providers (Gemini, Groq, OpenAI).

Includes automatic fallback: if the requested provider fails or has an
invalid API key, falls back to Gemini as the guaranteed-available provider.
"""

import logging
import os
from langchain_core.language_models.chat_models import BaseChatModel

# Provider specific imports
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)

# Fallback defaults (Gemini is the guaranteed working provider)
_FALLBACK_MODEL = "gemini-2.5-flash"
_FALLBACK_PROVIDER = "gemini"


def _create_llm(model_name: str, provider: str, temperature: float) -> BaseChatModel:
    """
    Internal: create a single LLM instance for the given provider.
    Only passes provider-appropriate arguments.
    """
    provider = provider.lower().strip()

    if provider == "gemini":
        return ChatGoogleGenerativeAI(
            model=model_name,
            temperature=temperature,
        )

    elif provider == "groq":
        return ChatGroq(
            model_name=model_name,
            temperature=temperature,
        )

    elif provider == "openai":
        return ChatOpenAI(
            model=model_name,
            temperature=temperature,
        )

    else:
        raise ValueError(f"Unsupported LLM provider: '{provider}'. Supported: ['gemini', 'groq', 'openai']")


def get_llm(
    model_name: str,
    provider: str,
    temperature: float = 0.0,
    **kwargs  # Accepted for backward compat but not forwarded (prevents cross-provider kwarg issues)
) -> BaseChatModel:
    """
    Instantiate the appropriate LangChain BaseChatModel.

    If the requested provider has an invalid/missing API key, automatically
    falls back to Gemini (gemini-2.5-flash) so the pipeline never hard-crashes.

    Args:
        model_name: The specific model ID (e.g., "gemini-2.5-flash", "llama-3.3-70b-versatile").
        provider: The provider name ("gemini", "groq", "openai").
        temperature: The sampling temperature.

    Returns:
        BaseChatModel: An instantiated LangChain chat model.
    """
    provider_lower = provider.lower().strip()

    # --- Key prefix validation (catches common misconfigurations) ---
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

    # --- Attempt primary provider ---
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
