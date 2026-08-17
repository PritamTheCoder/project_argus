"""
Project Argus - LLM factory tests (construction only, no network)

Verifies the proactive rate limiter is attached to every model and that the
Nemotron provider wires up / degrades correctly.
"""

import os
import pytest
from unittest.mock import patch

from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI

from src.utils.llm_factory import get_llm, _get_rate_limiter, _groq_keys, _next_groq_key


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


# ── Groq multi-key pool ──────────────────────────────────────────────────────

@patch.dict(os.environ, {"GROQ_API_KEY_A": "gsk_aaa", "GROQ_API_KEY_B": "gsk_bbb", "GROQ_API_KEY": ""}, clear=False)
def test_groq_pool_resolves_both_keys():
    assert _groq_keys() == ["gsk_aaa", "gsk_bbb"]


@patch.dict(os.environ, {"GROQ_API_KEY_A": "gsk_aaa", "GROQ_API_KEY_B": "gsk_bbb", "GROQ_API_KEY": ""}, clear=False)
def test_groq_pool_round_robins():
    # Consecutive selections alternate across the two pool indices.
    idxs = [_next_groq_key()[1] for _ in range(4)]
    assert set(idxs) == {0, 1}
    assert idxs[0] != idxs[1]  # alternates rather than repeating


@patch.dict(os.environ, {"GROQ_API_KEY_A": "gsk_aaa", "GROQ_API_KEY_B": "gsk_bbb", "GROQ_API_KEY": ""}, clear=False)
def test_groq_builds_rotate_keys_and_limiters():
    a = get_llm("llama-3.3-70b-versatile", "groq")
    b = get_llm("llama-3.3-70b-versatile", "groq")
    assert isinstance(a, ChatGroq) and isinstance(b, ChatGroq)
    # Different accounts → different keys AND independent rate-limiter buckets.
    assert a.groq_api_key.get_secret_value() != b.groq_api_key.get_secret_value()
    assert a.rate_limiter is not b.rate_limiter


@patch.dict(os.environ, {"GROQ_API_KEY_A": "", "GROQ_API_KEY_B": "", "GROQ_API_KEY": ""}, clear=False)
def test_groq_falls_back_to_gemini_without_keys():
    llm = get_llm("llama-3.3-70b-versatile", "groq")
    assert isinstance(llm, ChatGoogleGenerativeAI)


@patch.dict(os.environ, {"GROQ_API_KEY_A": "badprefix", "GROQ_API_KEY_B": "", "GROQ_API_KEY": ""}, clear=False)
def test_groq_falls_back_when_no_valid_prefix():
    llm = get_llm("llama-3.3-70b-versatile", "groq")
    assert isinstance(llm, ChatGoogleGenerativeAI)


def test_unknown_provider_raises():
    with pytest.raises(ValueError):
        # bypass key guards by going straight through _create_llm
        from src.utils.llm_factory import _create_llm
        _create_llm("x", "made_up_provider", 0.0)
