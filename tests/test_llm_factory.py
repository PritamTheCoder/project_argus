"""
Project Argus - LLM factory tests (construction only, no network)

Verifies the proactive rate limiter is attached to every model and that the
Nemotron provider wires up / degrades correctly.
"""

import pytest
from unittest.mock import patch

from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

from src.utils.llm_factory import get_llm, _get_rate_limiter


def test_gemini_model_has_rate_limiter():
    llm = get_llm("gemini-2.5-flash-lite", "gemini")
    assert isinstance(llm.rate_limiter, InMemoryRateLimiter)


def test_rate_limiter_shared_per_provider():
    a = get_llm("gemini-2.5-flash-lite", "gemini")
    b = get_llm("gemini-2.5-flash-lite", "gemini")
    # Same provider → same shared limiter instance (so RPM is enforced globally).
    assert a.rate_limiter is b.rate_limiter


def test_rate_limiter_distinct_across_providers():
    assert _get_rate_limiter("gemini") is not _get_rate_limiter("groq")


def test_rate_limiter_rps_matches_configured_rpm():
    from src.config import GEMINI_RPM
    limiter = _get_rate_limiter("gemini")
    assert limiter.requests_per_second == pytest.approx(GEMINI_RPM / 60.0)


def test_nemotron_constructs_with_key():
    from src.config import NVIDIA_API_NEMOTRON3_KEY
    if not NVIDIA_API_NEMOTRON3_KEY:
        pytest.skip("NVIDIA_API_NEMOTRON3_KEY not configured")
    llm = get_llm("nvidia/nemotron-3-ultra-550b-a55b", "nemotron")
    assert isinstance(llm, ChatOpenAI)
    assert isinstance(llm.rate_limiter, InMemoryRateLimiter)


@patch("src.utils.llm_factory.NVIDIA_API_NEMOTRON3_KEY", "")
def test_nemotron_falls_back_without_key():
    llm = get_llm("nvidia/nemotron-3-ultra-550b-a55b", "nemotron")
    # No key → graceful fallback to the default Gemini model.
    assert isinstance(llm, ChatGoogleGenerativeAI)


def test_unknown_provider_raises():
    with pytest.raises(ValueError):
        # bypass key guards by going straight through _create_llm
        from src.utils.llm_factory import _create_llm
        _create_llm("x", "made_up_provider", 0.0)
