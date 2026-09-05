"""
Project Argus - LLM factory tests (construction only, no network)

Verifies the proactive rate limiter is attached to every model and that the
Nemotron provider wires up / degrades correctly.
"""

import os
import pytest
from unittest.mock import patch

from src.utils.rate_limit import TokenAwareRateLimiter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI

from src.utils.llm_factory import get_llm, _get_rate_limiter, _groq_keys, _next_groq_key


def test_gemini_model_has_rate_limiter():
    llm = get_llm("gemini-2.5-flash-lite", "gemini")
    assert isinstance(llm.rate_limiter, TokenAwareRateLimiter)


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
    assert isinstance(llm.rate_limiter, TokenAwareRateLimiter)


@patch("src.utils.llm_factory.NVIDIA_API_NEMOTRON3_KEY", "")
def test_nemotron_falls_back_without_key():
    llm = get_llm("nvidia/nemotron-3-ultra-550b-a55b", "nemotron")
    # No key → graceful fallback to the default Gemini model.
    assert isinstance(llm, ChatGoogleGenerativeAI)


# ── Groq multi-key pool ──────────────────────────────────────────────────────

@patch.dict(os.environ, {"GROQ_API_KEY_A": "gsk_aaa", "GROQ_API_KEY_B": "gsk_bbb", "GROQ_API_KEY_C": "", "GROQ_API_KEY": ""}, clear=False)
def test_groq_pool_resolves_both_keys():
    assert _groq_keys() == ["gsk_aaa", "gsk_bbb"]


@patch.dict(os.environ, {"GROQ_API_KEY_A": "gsk_aaa", "GROQ_API_KEY_B": "gsk_bbb", "GROQ_API_KEY_C": "gsk_ccc", "GROQ_API_KEY": ""}, clear=False)
def test_groq_pool_resolves_three_keys():
    assert _groq_keys() == ["gsk_aaa", "gsk_bbb", "gsk_ccc"]


@patch.dict(os.environ, {"GROQ_API_KEY_A": "gsk_aaa", "GROQ_API_KEY_B": "gsk_bbb", "GROQ_API_KEY_C": "", "GROQ_API_KEY": ""}, clear=False)
def test_groq_pool_round_robins():
    idxs = [_next_groq_key()[1] for _ in range(4)]
    assert set(idxs) == {0, 1}
    assert idxs[0] != idxs[1]  # alternates rather than repeating


@patch.dict(os.environ, {"GROQ_API_KEY_A": "gsk_aaa", "GROQ_API_KEY_B": "gsk_bbb", "GROQ_API_KEY_C": "gsk_ccc", "GROQ_API_KEY": ""}, clear=False)
def test_groq_pool_round_robins_three_keys():
    idxs = [_next_groq_key()[1] for _ in range(6)]
    assert set(idxs) == {0, 1, 2}


@patch.dict(os.environ, {"GROQ_API_KEY_A": "gsk_aaa", "GROQ_API_KEY_B": "gsk_bbb", "GROQ_API_KEY_C": "", "GROQ_API_KEY": ""}, clear=False)
def test_groq_builds_rotate_keys_and_limiters():
    a = get_llm("llama-3.3-70b-versatile", "groq")
    b = get_llm("llama-3.3-70b-versatile", "groq")
    assert isinstance(a, ChatGroq) and isinstance(b, ChatGroq)
    # Different accounts → different keys AND independent rate-limiter buckets.
    assert a.groq_api_key.get_secret_value() != b.groq_api_key.get_secret_value()
    assert a.rate_limiter is not b.rate_limiter


@patch.dict(os.environ, {"GROQ_API_KEY_A": "", "GROQ_API_KEY_B": "", "GROQ_API_KEY_C": "", "GROQ_API_KEY": ""}, clear=False)
def test_groq_falls_back_to_gemini_without_keys():
    llm = get_llm("llama-3.3-70b-versatile", "groq")
    assert isinstance(llm, ChatGoogleGenerativeAI)


@patch.dict(os.environ, {"GROQ_API_KEY_A": "badprefix", "GROQ_API_KEY_B": "", "GROQ_API_KEY_C": "", "GROQ_API_KEY": ""}, clear=False)
def test_groq_falls_back_when_no_valid_prefix():
    llm = get_llm("llama-3.3-70b-versatile", "groq")
    assert isinstance(llm, ChatGoogleGenerativeAI)


def test_unknown_provider_raises():
    with pytest.raises(ValueError):
        # bypass key guards by going straight through _create_llm
        from src.utils.llm_factory import _create_llm
        _create_llm("x", "made_up_provider", 0.0)


def test_writer_ladder_excludes_nemotron():
    """Nemotron's slow failover on a 503 can blow a case's timeout on its own."""
    from src.config import WRITER_FALLBACK_CHAIN
    assert "nemotron" not in WRITER_FALLBACK_CHAIN


def test_groq_limiter_gates_on_tokens():
    """Groq's free tier rejects on tokens/minute before requests/minute, so the
    token gate must be armed for it."""
    from src.config import GROQ_TPM
    limiter = _get_rate_limiter("groq#0", 30, GROQ_TPM)
    assert limiter.gates_on_tokens


# ── Groq account pool as failover ────────────────────────────────────────────
# TPD is per organization, so separate Groq accounts hold separate daily budgets.
# A 429 on one key must try the siblings before abandoning Groq for another
# provider, or the remaining accounts' quota goes unused for the rest of the run.

def test_groq_pool_yields_one_rung_per_account():
    from src.utils.llm_factory import _groq_pool_rungs
    with patch("src.utils.llm_factory._groq_keys", return_value=["gsk_a", "gsk_b", "gsk_c"]):
        rungs = _groq_pool_rungs("m", "groq")
    assert len(rungs) == 3
    assert {r[2] for r in rungs} == {0, 1, 2}, "every account should get a rung"


def test_groq_pool_rungs_start_at_the_round_robin_position():
    """Load still spreads across accounts; only the failover order is pinned."""
    from src.utils.llm_factory import _groq_pool_rungs
    with patch("src.utils.llm_factory._groq_keys", return_value=["a", "b", "c"]):
        first = _groq_pool_rungs("m", "groq")[0][2]
        second = _groq_pool_rungs("m", "groq")[0][2]
    assert first != second


def test_single_groq_key_keeps_round_robin_behaviour():
    from src.utils.llm_factory import _groq_pool_rungs
    with patch("src.utils.llm_factory._groq_keys", return_value=["only"]):
        assert _groq_pool_rungs("m", "groq") == [("m", "groq", None)]


def test_non_groq_provider_gets_a_single_rung():
    from src.utils.llm_factory import _groq_pool_rungs
    assert _groq_pool_rungs("m", "gemini") == [("m", "gemini", None)]


def test_pinned_key_index_is_used_instead_of_round_robin():
    from src.utils.llm_factory import _create_llm
    keys = ["gsk_aaa", "gsk_bbb", "gsk_ccc"]
    with patch("src.utils.llm_factory._groq_keys", return_value=keys):
        llm = _create_llm("m", "groq", 0.0, groq_key_index=2)
    assert llm.groq_api_key.get_secret_value() == "gsk_ccc"


def test_each_account_gets_its_own_rate_limiter():
    """Separate accounts have separate budgets, so they must not share a bucket."""
    from src.utils.llm_factory import _create_llm
    keys = ["gsk_aaa", "gsk_bbb"]
    with patch("src.utils.llm_factory._groq_keys", return_value=keys):
        a = _create_llm("m", "groq", 0.0, groq_key_index=0)
        b = _create_llm("m", "groq", 0.0, groq_key_index=1)
    assert a.rate_limiter is not b.rate_limiter


def test_groq_does_not_retry_a_quota_exhausted_key_in_sdk():
    """The client honours retry-after and would sleep re-asking a quota'd key.
    The ladder's next rung is a different account, so it should fail over
    immediately instead."""
    from src.utils.llm_factory import _create_llm
    with patch("src.utils.llm_factory._groq_keys", return_value=["gsk_a"]):
        llm = _create_llm("m", "groq", 0.0)
    assert llm.max_retries == 0


# ── Per-rung failure logging ─────────────────────────────────────────────────
# A rung that fails before a later rung succeeds never otherwise surfaces its
# error text: httpx's own request log shows only method/URL/status, and
# application code only sees the *last* rung's exception if every rung fails.

def test_rung_failure_logger_fires_on_a_masked_failure(caplog):
    """The exact case that was invisible: rung 1 fails, rung 2 succeeds, and
    rung 1's real error must still be logged somewhere."""
    import logging as _logging
    from langchain_core.language_models.fake_chat_models import (
        FakeChatModel, FakeMessagesListChatModel,
    )
    from langchain_core.messages import AIMessage
    from src.utils.llm_factory import RungFailureLogger

    class BrokenChatModel(FakeChatModel):
        def _generate(self, *a, **kw):
            raise RuntimeError("simulated 400 Bad Request: bad payload")

    bad = BrokenChatModel()
    good = FakeMessagesListChatModel(responses=[AIMessage(content="fallback ok")])
    chain = bad.with_fallbacks([good])

    with caplog.at_level(_logging.WARNING, logger="src.utils.llm_factory"):
        result = chain.invoke("hi", config={"callbacks": [RungFailureLogger()]})

    assert result.content == "fallback ok"
    assert any("bad payload" in r.message for r in caplog.records)


def test_default_callbacks_always_include_the_failure_logger():
    from src.utils.llm_factory import _default_callbacks, RungFailureLogger, _get_rate_limiter
    limiter = _get_rate_limiter("gemini")
    names = [type(c).__name__ for c in _default_callbacks(limiter)]
    assert "RungFailureLogger" in names


def test_factory_built_llm_carries_the_failure_logger():
    from src.utils.llm_factory import get_llm
    llm = get_llm("openai/gpt-oss-120b", "groq")
    names = [type(c).__name__ for c in llm.callbacks]
    assert "RungFailureLogger" in names
