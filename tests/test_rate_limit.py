"""
Project Argus - Token-aware rate limiter tests
"""

import time

import pytest
from langchain_core.outputs import ChatGeneration, LLMResult
from langchain_core.messages import AIMessage

from src.utils.rate_limit import (
    TokenAwareRateLimiter,
    TokenUsageReporter,
    extract_total_tokens,
)


def _limiter(**kw) -> TokenAwareRateLimiter:
    defaults = dict(requests_per_minute=600, tokens_per_minute=1000, burst=8,
                    default_call_tokens=100)
    defaults.update(kw)
    return TokenAwareRateLimiter(**defaults)


# ── Token gating ─────────────────────────────────────────────────────────────

def test_token_budget_blocks_when_exhausted():
    """A provider with a token ceiling stops issuing calls once it is spent."""
    lim = _limiter(tokens_per_minute=300, default_call_tokens=100)
    assert lim.acquire(blocking=False)
    assert lim.acquire(blocking=False)
    assert lim.acquire(blocking=False)
    # 3 x 100 reserved against a 300-token minute: the next call must wait.
    assert not lim.acquire(blocking=False)


def test_requests_still_gated_when_no_token_ceiling():
    """tokens_per_minute=0 disables the token gate but not the request gate."""
    lim = _limiter(requests_per_minute=60, tokens_per_minute=0, burst=2)
    assert lim.acquire(blocking=False)
    assert lim.acquire(blocking=False)
    assert not lim.acquire(blocking=False)


def test_no_token_ceiling_does_not_gate_on_tokens():
    lim = _limiter(tokens_per_minute=0, burst=100)
    for _ in range(50):
        assert lim.acquire(blocking=False)
    assert not lim.gates_on_tokens


def test_token_bucket_refills_over_time():
    lim = _limiter(tokens_per_minute=6000, default_call_tokens=100)  # 100 tok/s
    for _ in range(8):
        lim.acquire(blocking=False)
    while lim.acquire(blocking=False):
        pass
    time.sleep(0.25)
    # Refill alone should restore enough budget for at least one more call.
    assert lim.acquire(blocking=False)


# ── Usage feedback ───────────────────────────────────────────────────────────

def test_underestimated_call_is_debited_not_ignored():
    """A call costing more than reserved must consume the extra budget, or the
    limiter would drift over the real ceiling exactly as RPM-only did."""
    lim = _limiter(tokens_per_minute=1000, default_call_tokens=100)
    lim.acquire(blocking=False)
    lim.record_usage(900)  # reserved 100, actually cost 900
    assert not lim.acquire(blocking=False)


def test_cheaper_call_refunds_budget():
    lim = _limiter(tokens_per_minute=300, default_call_tokens=100)
    for _ in range(3):
        lim.acquire(blocking=False)
    assert not lim.acquire(blocking=False)
    lim.record_usage(10)  # reserved 100, cost 10 → 90 back
    assert lim.acquire(blocking=False)


def test_average_call_size_tracks_observed_usage():
    lim = _limiter(tokens_per_minute=100000, default_call_tokens=100)
    for _ in range(20):
        lim.record_usage(5000)
    assert lim._avg_call_tokens == pytest.approx(5000, rel=0.05)


def test_record_usage_is_a_noop_without_a_token_ceiling():
    lim = _limiter(tokens_per_minute=0, burst=4)
    lim.record_usage(10_000)
    for _ in range(4):
        assert lim.acquire(blocking=False)


def test_reserve_never_exceeds_a_full_minute_of_budget():
    """A single huge call must not reserve more than the bucket can ever hold,
    which would block every subsequent call forever."""
    lim = _limiter(tokens_per_minute=1000, default_call_tokens=100)
    for _ in range(50):
        lim.record_usage(50_000)  # average climbs far above the ceiling
    lim._tok_tokens = 1000.0      # a full bucket
    assert lim.acquire(blocking=False)


# ── Usage extraction across provider shapes ──────────────────────────────────

def test_extracts_usage_from_openai_style_llm_output():
    res = LLMResult(generations=[[]], llm_output={"token_usage": {"total_tokens": 1234}})
    assert extract_total_tokens(res) == 1234


def test_extracts_usage_from_message_metadata():
    """Gemini reports usage on the message, not in llm_output."""
    msg = AIMessage(content="hi", usage_metadata={
        "input_tokens": 10, "output_tokens": 5, "total_tokens": 15,
    })
    res = LLMResult(generations=[[ChatGeneration(message=msg)]], llm_output={})
    assert extract_total_tokens(res) == 15


def test_missing_usage_reports_zero():
    res = LLMResult(generations=[[]], llm_output=None)
    assert extract_total_tokens(res) == 0


def test_reporter_feeds_usage_into_the_limiter():
    lim = _limiter(tokens_per_minute=1000, default_call_tokens=100)
    lim.acquire(blocking=False)
    reporter = TokenUsageReporter(lim)
    reporter.on_llm_end(
        LLMResult(generations=[[]], llm_output={"token_usage": {"total_tokens": 900}})
    )
    assert not lim.acquire(blocking=False)
