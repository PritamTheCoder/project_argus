"""
Project Argus - Token-aware client-side rate limiting.

LangChain's InMemoryRateLimiter only models requests per minute. Free-tier Groq
rejects on *tokens* per minute (8000 for gpt-oss-120b) long before requests, so
an RPM-only limiter happily lets a burst of large prompts through and the run
pays for them in 429s and duplicate fallback calls.

This limiter gates on both. The rate-limiter interface never sees the request
payload, so it cannot weigh a call before making it; instead it reserves a
running average of recent call sizes and corrects that average against real
usage as responses come back.
"""

import asyncio
import logging
import threading
import time
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from langchain_core.rate_limiters import BaseRateLimiter

logger = logging.getLogger(__name__)

# Weight given to the newest observation when updating the average call size.
# Low enough that one outlier batch doesn't stall the bucket for everyone.
_EWMA_ALPHA = 0.3

# How long a blocked acquire sleeps between checks.
_POLL_SECONDS = 0.1


class TokenAwareRateLimiter(BaseRateLimiter):
    """Allows a call only when both the request and token budgets permit it.

    Two token buckets refill continuously: one in requests, one in tokens. A
    call must take a slot from each. ``record_usage`` reports what a call
    actually cost, which both trues up the bucket and sharpens the estimate
    used to reserve for the next one.
    """

    def __init__(
        self,
        requests_per_minute: float,
        tokens_per_minute: int = 0,
        burst: int = 8,
        default_call_tokens: int = 2000,
    ) -> None:
        self._rps = max(requests_per_minute / 60.0, 0.05)
        self._req_capacity = float(burst)
        self._req_tokens = float(burst)

        # tokens_per_minute == 0 means this provider has no token ceiling.
        self._tpm = max(int(tokens_per_minute), 0)
        self._tps = self._tpm / 60.0
        # A full minute of budget, so a burst can spend ahead and then wait.
        self._tok_capacity = float(self._tpm)
        self._tok_tokens = float(self._tpm)

        self._avg_call_tokens = float(max(default_call_tokens, 1))
        self._last = time.monotonic()
        self._lock = threading.Lock()

    @property
    def requests_per_second(self) -> float:
        return self._rps

    @property
    def gates_on_tokens(self) -> bool:
        """True when a token ceiling is configured for this provider."""
        return bool(self._tpm)

    # ── bucket maintenance ───────────────────────────────────────────────────

    def _refill(self) -> None:
        """Add the budget accrued since the last call. Caller holds the lock."""
        now = time.monotonic()
        elapsed = now - self._last
        if elapsed <= 0:
            return
        self._last = now
        self._req_tokens = min(self._req_capacity, self._req_tokens + elapsed * self._rps)
        if self._tpm:
            self._tok_tokens = min(self._tok_capacity, self._tok_tokens + elapsed * self._tps)

    def _reserve(self) -> float:
        """Tokens to hold for the next call, capped so one call can't deadlock
        the bucket by reserving more than a full minute of budget."""
        if not self._tpm:
            return 0.0
        return min(self._avg_call_tokens, self._tok_capacity)

    def _try_acquire(self) -> bool:
        """Take one slot from each bucket if both allow it."""
        with self._lock:
            self._refill()
            if self._req_tokens < 1.0:
                return False
            need = self._reserve()
            if self._tpm and self._tok_tokens < need:
                return False
            self._req_tokens -= 1.0
            self._tok_tokens -= need
            return True

    # ── BaseRateLimiter interface ────────────────────────────────────────────

    def acquire(self, *, blocking: bool = True) -> bool:
        if not blocking:
            return self._try_acquire()
        while not self._try_acquire():
            time.sleep(_POLL_SECONDS)
        return True

    async def aacquire(self, *, blocking: bool = True) -> bool:
        if not blocking:
            return self._try_acquire()
        while not self._try_acquire():
            await asyncio.sleep(_POLL_SECONDS)
        return True

    # ── usage feedback ───────────────────────────────────────────────────────

    def record_usage(self, total_tokens: int) -> None:
        """Reconcile a finished call against what was reserved for it."""
        if not self._tpm or total_tokens <= 0:
            return
        with self._lock:
            self._refill()
            # Charge the difference between the true cost and what was held.
            # A call cheaper than the estimate refunds; a costlier one is debited,
            # so an underestimate cannot silently overrun the ceiling.
            self._tok_tokens = max(
                -self._tok_capacity,
                min(self._tok_capacity, self._tok_tokens + self._reserve() - total_tokens),
            )
            self._avg_call_tokens = (
                _EWMA_ALPHA * total_tokens + (1 - _EWMA_ALPHA) * self._avg_call_tokens
            )


def extract_total_tokens(response: LLMResult) -> int:
    """Total tokens for a finished call, or 0 if the provider didn't report any.

    Providers disagree on where usage lands: OpenAI-compatible backends (Groq,
    NIM) put it in ``llm_output``, while Gemini attaches it to the message.
    """
    usage = (response.llm_output or {}).get("token_usage") or {}
    total = usage.get("total_tokens") or 0
    if total:
        return int(total)

    for generations in response.generations:
        for gen in generations:
            message = getattr(gen, "message", None)
            meta = getattr(message, "usage_metadata", None) or {}
            if meta.get("total_tokens"):
                return int(meta["total_tokens"])
    return 0


class TokenUsageReporter(BaseCallbackHandler):
    """Feeds finished calls' token usage back into their provider's limiter."""

    def __init__(self, limiter: TokenAwareRateLimiter) -> None:
        self._limiter = limiter

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        total = extract_total_tokens(response)
        if total:
            self._limiter.record_usage(total)
